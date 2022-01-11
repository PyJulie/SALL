import numpy as np
from scipy import misc
from keras.applications.inception_v3 import preprocess_input
def generator(data,batch_size,base_image_dir,dic,dic_image):
    while True:
        batch = np.random.choice(data, batch_size)
        x,y = [],[]
        y1, y2, y3 = [],[],[]
        for img in batch:
                x.append(misc.imread(base_image_dir+'/'+dic[img[0]]+'/'+img))

                test_y = dic_image[img]
                dr_y = [float(x) for x in test_y.split(' ')[:5]]
                amd_y = [float(x) for x in test_y.split(' ')[5:9]]
                arter_y = [float(x) for x in test_y.split(' ')[9:]]
                y1.append(dr_y)
                y2.append(amd_y)
                y3.append(arter_y)
        y1 = np.array(y1)
        y2 = np.array(y2)
        y3 = np.array(y3)        
        x = preprocess_input(np.array(x).astype(float))
        y = [y1,y2,y3]
        yield x,y
        x,y = [],[]
        y1, y2, y3 = [],[],[]
