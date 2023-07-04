import os
import sys
import json
import subprocess
from extract_sfm import extract_all_to_dir
from normalize_cam_dict import normalize_cam_dict
from run_colmap import run_sfm, prepare_mvs, extract_all_to_dir, normalize_cam_dict
import random
import shutil
from PIL import Image

def crop_image( src_path, dst_path ):
    img = Image.open( src_path )
    img_crop = img.crop((0, 15, 1080, 1935))
    img_crop.save( dst_path )
    # shutil.copy( img_src_path, img_dst_path )

def save_camera_path_data(out_dir, scene, json_data):
    scene_out_dir = os.path.join(out_dir, scene)
    camera_path_dir = os.path.join( scene_out_dir, 'camera_path' )
    intrinsics_dir = os.path.join(camera_path_dir, 'intrinsics')
    pose_dir = os.path.join(camera_path_dir, 'pose')
    
    os.makedirs(camera_path_dir, exist_ok=True)
    os.makedirs(intrinsics_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    for image_name, image_pose in json_data.items():
        image_txt_name = os.path.splitext( image_name )[0] + '.txt'
        intrinisics_path = os.path.join( intrinsics_dir, image_txt_name )
        pose_path = os.path.join( pose_dir, image_txt_name )
        with open( intrinisics_path, 'wt' ) as fd:
            fd.write( ' '.join([str(i) for i in image_pose['K']]) )
        with open( pose_path, 'wt' ) as fd:
            fd.write( ' '.join([str(i) for i in image_pose['W2C']]) )

def save_train_data(out_dir, scene, json_data, train_keys, mode='train'):
    scene_out_dir = os.path.join(out_dir, scene)
    train_dir = os.path.join( scene_out_dir, mode )
    intrinsics_dir = os.path.join(train_dir, 'intrinsics')
    pose_dir = os.path.join(train_dir, 'pose')
    rgb_dir = os.path.join(train_dir, 'rgb')
    mvs_dir = os.path.join(out_dir, 'mvs')
    undistorted_img_dir = os.path.join(mvs_dir, 'images')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(intrinsics_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)

    for image_name in train_keys:
        image_pose = json_data[image_name]
        txt_name = os.path.splitext( image_name )[0]
        txt_name = txt_name.split('_')[1]
        txt_name = '{}_{}.txt'.format( scene, txt_name )
        
        intrinisics_path = os.path.join( intrinsics_dir, txt_name )
        pose_path = os.path.join( pose_dir, txt_name )
        # print( intrinisics_path )

        with open( intrinisics_path, 'wt' ) as fd:
            fd.write( ' '.join([str(i) for i in image_pose['K']]) )
        with open( pose_path, 'wt' ) as fd:
            fd.write( ' '.join([str(i) for i in image_pose['W2C']]) )

        image_name_ex = image_name.replace('frame', scene)
        img_src_path = os.path.join(undistorted_img_dir, image_name)
        img_dst_path = os.path.join(rgb_dir, image_name_ex)
        print( '{} => {}'.format(img_src_path, img_dst_path) )
        crop_image( img_src_path, img_dst_path )
        
def transform_to_nerfplus_format(out_dir, scene):
    # 从colmap结果转换到最终的目录中
    if os.path.isdir( out_dir ) == False:
        return False 

    scene_out_dir = os.path.join(out_dir, scene)
    os.makedirs(scene_out_dir, exist_ok=True)

    mvs_dir = os.path.join(out_dir, 'mvs')
    undistorted_img_dir = os.path.join(mvs_dir, 'images')
    result_json_path = os.path.join(out_dir, 'posed_images/kai_cameras_normalized.json')
    
    train_data_list = []
    test_data_list = []

    with open( result_json_path ) as json_fp:
        json_data = json.load( json_fp )
        # print( json_data )
        total_keys = list(json_data.keys())
        train_data_len = test_data_len = len( total_keys ) // 2
        if train_data_len > 100:
            train_data_len = test_data_len = 100
        
        random.shuffle( total_keys )
        # print( total_keys )
        train_keys = total_keys[0:train_data_len]
        test_keys = total_keys[train_data_len:train_data_len+test_data_len]
        # print( train_keys )
        # print( test_keys )

        save_camera_path_data(out_dir, scene, json_data)
        save_train_data( out_dir, scene, json_data, train_keys, 'train' )
        save_train_data( out_dir, scene, json_data, test_keys, 'test' )

        scene_out_dir = os.path.join(out_dir, scene)
        test_dir = os.path.join( scene_out_dir, 'test' )
        validate_dir_link = os.path.join( scene_out_dir, 'validation' )
        os.symlink(test_dir, validate_dir_link)


# 从nerfstudio产出的数据，直接转换到nerfplusplus的数据格式
# in_dir: nerfstudio生成的目录
# out_dir： nerfplus结果目录
# scene： 对应的nerfplus场景标识（就是一个目录名而已）
def generate_from_nerfstudio(in_dir, out_dir, scene):
    # verify input directory
    img_dir = os.path.join( in_dir, 'images' )
    if os.path.exists( img_dir ) == False:
        print('No image directory found!!!')
        return False
    
    ns_colmap_dir = os.path .join( in_dir, 'colmap' )
    if os.path.exists( ns_colmap_dir ) == False:
        print('No colmap directory found!!!')
        return False
    ns_db_file = os.path.join(ns_colmap_dir, 'database.db')
    ns_sparse_dir = os.path.join(ns_colmap_dir, 'sparse')
    if os.path.exists( ns_db_file ) == False or os.path.exists(ns_sparse_dir) == False:
        print('No colmap result found!!!')
        return False

    # create output directory and make soft link to image directory
    os.makedirs(out_dir, exist_ok=True)
    sfm_dir = os.path.join(out_dir, 'sfm')
    os.makedirs(sfm_dir, exist_ok=True)

    img_dir_link = os.path.join(sfm_dir, 'images')
    if os.path.exists(img_dir_link):
        os.remove(img_dir_link)
    os.symlink(img_dir, img_dir_link)

    # copy sfm result from nerfstudio
    shutil.copy( ns_db_file, os.path.join(sfm_dir, 'database.db') )
    shutil.copytree( ns_sparse_dir, os.path.join( sfm_dir, 'sparse' ) )
    sparse_dir = os.path.join( sfm_dir, 'sparse', '0' )

    # undistort images
    mvs_dir = os.path.join(out_dir, 'mvs')
    os.makedirs(mvs_dir, exist_ok=True)
    prepare_mvs(img_dir, sparse_dir, mvs_dir)

    # extract camera parameters and undistorted images
    os.makedirs(os.path.join(out_dir, 'posed_images'), exist_ok=True)
    extract_all_to_dir(os.path.join(mvs_dir, 'sparse'), os.path.join(out_dir, 'posed_images'))
    undistorted_img_dir = os.path.join(mvs_dir, 'images')
    posed_img_dir_link = os.path.join(out_dir, 'posed_images/images')
    if os.path.exists(posed_img_dir_link):
        os.remove(posed_img_dir_link)
    os.symlink(undistorted_img_dir, posed_img_dir_link)

    # normalize average camera center to origin, and put all cameras inside the unit sphere
    normalize_cam_dict(os.path.join(out_dir, 'posed_images/kai_cameras.json'),
                       os.path.join(out_dir, 'posed_images/kai_cameras_normalized.json'))

    # transform json result to nerfplus format 
    transform_to_nerfplus_format(out_dir, scene)

    return True

def main(img_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    #### run sfm
    sfm_dir = os.path.join(out_dir, 'sfm')
    os.makedirs(sfm_dir, exist_ok=True)

    img_dir_link = os.path.join(sfm_dir, 'images')
    if os.path.exists(img_dir_link):
        os.remove(img_dir_link)
    os.symlink(img_dir, img_dir_link)

    db_file = os.path.join(sfm_dir, 'database.db')
    # run_sift_matching(img_dir, db_file, remove_exist=False)
    sparse_dir = os.path.join(sfm_dir, 'sparse')
    os.makedirs(sparse_dir, exist_ok=True)
    run_sfm(img_dir, db_file, sparse_dir)

    # undistort images
    mvs_dir = os.path.join(out_dir, 'mvs')
    os.makedirs(mvs_dir, exist_ok=True)
    prepare_mvs(img_dir, sparse_dir, mvs_dir)

    # extract camera parameters and undistorted images
    os.makedirs(os.path.join(out_dir, 'posed_images'), exist_ok=True)
    extract_all_to_dir(os.path.join(mvs_dir, 'sparse'), os.path.join(out_dir, 'posed_images'))
    undistorted_img_dir = os.path.join(mvs_dir, 'images')
    posed_img_dir_link = os.path.join(out_dir, 'posed_images/images')
    if os.path.exists(posed_img_dir_link):
        os.remove(posed_img_dir_link)
    os.symlink(undistorted_img_dir, posed_img_dir_link)

    # normalize average camera center to origin, and put all cameras inside the unit sphere
    normalize_cam_dict(os.path.join(out_dir, 'posed_images/kai_cameras.json'),
                       os.path.join(out_dir, 'posed_images/kai_cameras_normalized.json'))

in_dir = '/home/wanglei19/workspace/data/nerf/211959'
out_dir = '/home/wanglei19/workspace/data/nerf/211959_np'
scene = 'cartoon'
generate_from_nerfstudio(in_dir, out_dir, scene)
