import numpy as np

embed1_path = r'/home/harsh/AI-Projects/person-detection/img2_embeddings.npy'
embed2_path = r'/home/harsh/AI-Projects/AI-Assistant/Face_Recognition/img_embeddings.npy'

embed1 = np.load(embed1_path)
embed2 = np.load(embed2_path)

def calc_dist(embedding1, embedding2):

    result = np.sqrt(np.sum(np.square(embedding2 - embedding1)))
    print('distance : ', result)
    return result

result = calc_dist(embedding1=embed1, embedding2= embed2)
print(result)