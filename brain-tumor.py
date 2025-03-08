import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


input_folder_yes= 'braintumor/yes'
input_folder_no= 'braintumor/no'
gray_folder = 'output5'

os.makedirs(gray_folder, exist_ok=True)


from sklearn.cluster import KMeans

def process_images(folder,label):
    
    image_data = []
    for name in os.listdir(folder):
        img_path = os.path.join(folder, name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        
        graycode_image = np.bitwise_xor(image,image >> 1)
        pixels = graycode_image.reshape(-1,1)
        
        
        
        
        
        kmeans =KMeans(n_clusters=5,n_init='auto')
        kmeans.fit(pixels)
        
            
        
        
        
        
        segmented_img = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)
        gray_path = os.path.join(gray_folder, f"{label}_{name}")
        cv2.imwrite(gray_path, segmented_img)
        image_data.append(pixels.mean())
        
        
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.title("Original")

        plt.subplot(1, 3, 2)
        plt.imshow(graycode_image, cmap="gray")
        plt.axis("off")
        plt.title("Gray Code Conversion")

        plt.subplot(1, 3, 3)
        plt.imshow(segmented_img, cmap="gray")
        plt.axis("off")
        plt.title("K-Means Clustering")

        plt.show()
        
        
    

    return image_data



var_data=process_images(input_folder_yes, "yes")
yok_data=process_images(input_folder_no, "no")




plt.figure(figsize=(8, 5))
plt.hist(var_data, bins=30, alpha=0.7, label="yes")
plt.hist(yok_data, bins=30, alpha=0.7, label="no")
plt.legend()
plt.xlabel("Mean Pixel Value")
plt.ylabel("Frequency")
plt.show()




