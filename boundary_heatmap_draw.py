import cv2 
import os
import timeit 
import glob
import numpy as np


"""def sketch2heatmap(img_path = "processed_lrs2_ours/lrs2_sketch_jaw/5535415699068794046/00001/0.png"):
    # preprocess cancel comment
    # after preprocess, change the next one

    img = cv2.imread(img_path, 0)
    sigma = 1.0

    heatmap = np.uint8(img)
    heatmap = cv2.distanceTransform(heatmap, cv2.DIST_L2, 5)
    heatmap = np.float32(np.array(heatmap))

    heatmap[heatmap < 3. * sigma] = \
                np.exp(-heatmap[heatmap < 3 * sigma] *
                    heatmap[heatmap < 3 * sigma] / 2. * sigma * sigma)

    heatmap[heatmap >= 3. * sigma] = 0.
    heatmap = np.array(heatmap) # (128*128)

    # end_time = timeit.default_timer()
    return heatmap"""

def sketch2heatmap(sketch):
    # 单通道
    # start_time = timeit.default_timer()
    # 推理的时候把这个打开

    sigma = 1.0

    heatmap = np.uint8(sketch)
    heatmap = cv2.distanceTransform(heatmap, cv2.DIST_L2, 5)
    heatmap = np.float32(np.array(heatmap))

    heatmap[heatmap < 3. * sigma] = \
                np.exp(-heatmap[heatmap < 3 * sigma] *
                    heatmap[heatmap < 3 * sigma] / 2. * sigma * sigma)

    heatmap[heatmap >= 3. * sigma] = 0.
    heatmap = np.array(heatmap) # (128*128)

    # end_time = timeit.default_timer()
    # print("sketch转热力图耗时为:", end_time - start_time)
    return heatmap

def heatmap_visualize(heatmap):
    # heatmap:np.array   value 0~1
    heatmap_visual = np.uint8(heatmap*255)
    heat_img = cv2.applyColorMap(heatmap_visual, cv2.COLORMAP_HOT)
    # cv2.imwrite('testtttt1.jpg', heat_img) 

    return heat_img
    # np.save('test.npy', heatmap)
    
def heatmap_generate_visualize(root_path = 'processed_lrs2_ours/lrs2_sketch_whole'):
    print('looking up sketch.... ')
    sketch_list = glob.glob(root_path + '/*/*/*.png')
    print('total sketch :', len(sketch_list))
    i = 0
    for path in sketch_list:
        i += 1
        if i % 1000 == 0:
            print(i)
        heatmap = sketch2heatmap(img_path=path)
        heat_img = heatmap_visualize(heatmap)
        
        heatmap_out_dir = os.path.join(root_path, '/'.join(path[:-4].split('/')[-3:-1])).replace('sketch', 'heatmap')
        os.makedirs(heatmap_out_dir, exist_ok=True)
        np.save(os.path.join(heatmap_out_dir, str(path[:-4].split('/')[-1]))+'.npy', heatmap)
        
        heat_img_out_dir = os.path.join(root_path, '/'.join(path[:-4].split('/')[-3:-1])).replace('sketch', 'heat_img')
        os.makedirs(heat_img_out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(heat_img_out_dir, str(path[:-4].split('/')[-1]))+'.png', heat_img)
        
    """for path in sketch_list:
        heatmap1 = np.load(path)
        heatmap2 = np.load(path.replace("upper", "lip"))
        heatmap3 = np.load(path.replace("upper", "jaw"))
        heatmap = heatmap1 + heatmap2 + heatmap3
        
        heat_img = heatmap_visualize(heatmap)
        
        heatmap_out_dir = os.path.join(root_path, '/'.join(path[:-4].split('/')[-3:-1])).replace('upper', 'whole')
        os.makedirs(heatmap_out_dir, exist_ok=True)
        np.save(os.path.join(heatmap_out_dir, str(path[:-4].split('/')[-1]))+'.npy', heatmap)
        
        heat_img_out_dir = os.path.join(root_path, '/'.join(path[:-4].split('/')[-3:-1])).replace('heatmap_upper', 'heat_img_whole')
        os.makedirs(heat_img_out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(heat_img_out_dir, str(path[:-4].split('/')[-1]))+'.png', heat_img)"""
    
if __name__ == "__main__":

    # heatmap_visualize(heatmap1)
    heatmap_generate_visualize(root_path = 'processed_lrs2_ours/lrs2_sketch_upper')
    print("done")
    # heatmap_generate_visualize(root_path = 'processed_hdtf/hdtf_sketch_upper')
    # print("done")
    # heatmap_generate_visualize(root_path = 'processed_hdtf/hdtf_sketch_whole_new')
    # print("done")
