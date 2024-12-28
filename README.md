# **Image Similarity Search: An Alternative to Google Lens**

## **Objective**
The goal of this project is to develop an alternative to Google Lens by implementing and comparing multiple approaches for image similarity search. Models like Yolo,Retinanet etc(using these models is straightforward so we can focus on image similarity)could be used to draw bounding boxes around main object in the image and then Image similarity models could be deployed above that to make something like google lens, then we proceed to Image similarity based on that main subject in image. The methods which could be used for image similarity include CNN-based models (ResNet-50), Vision Transformers (ViT), Autoencoders, Siamese Networks, and SIFT. Additionally, we discuss hashing-based methods and tools like FAISS and Spotify Annoy for vector similarity search, which could be utilized for larger datasets.

---
## **How to Run**
- Clone the Repository
- Install dependencies: pip install -r requirements.txt
- run the notebook for each approach
- Have also added links to my google colabs in links_to_notebooks.txt
---
## **Implemented Methods**

### **1. CNN-Based Feature Extraction (ResNet-50)**
#### **Architecture**
ResNet-50 is a convolutional neural network with 50 layers, utilizing skip connections to address the vanishing gradient problem. It is pre-trained on ImageNet and fine-tuned on the project dataset.

#### **Process**
- Extracted feature embeddings from the penultimate layer of ResNet-50.
- Used cosine similarity to compute similarity between images in the embedding space.

#### **Advantages**
- Pre-trained weights provide robust feature extraction.
- Suitable for datasets with diverse image classes.
  
---

### **2. Vision Transformers (ViT)**
#### **Architecture**
ViT splits images into patches and processes them as sequences using transformer encoders. It is highly effective for large datasets.

#### **Process**
- Pre-trained ViT model was fine-tuned on the dataset.
- Extracted embeddings from the final transformer layer for similarity computation.

#### **Advantages**
- State-of-the-art performance on vision tasks.
- Handles long-range dependencies well.

---

### **3. Autoencoder-Based Similarity Search**
#### **Architecture**
An autoencoder compresses input images into a latent space and reconstructs them. The latent space embeddings are used for similarity search.

#### **Process**
- Trained a convolutional autoencoder on the dataset.
- Compared images using cosine similarity in the latent space.

#### **Advantages**
- Simple and interpretable.
- Effective for smaller datasets.
---

### **4. Siamese Neural Networks**
#### **Architecture**
A Siamese network consists of two identical sub-networks sharing weights. It learns a similarity metric by comparing pairs of images.

#### **Process**
- Trained the network with contrastive loss.
- Used the learned metric to compute similarity scores.

#### **Advantages**
- Tailored for similarity tasks.
- Effective even with limited data.

---

### **5. SIFT (Scale-Invariant Feature Transform)**
#### **Process**
- Extracted key points and descriptors from images using SIFT.
- Matched descriptors using brute-force matching.

#### **Advantages**
- Does not require training.
- Effective for small datasets and specific tasks.

---

## **Additional Discussion**

### **6. Hashing-Based Methods**
Hashing converts high-dimensional data into compact binary codes for fast similarity search. Examples include:
- Locality-Sensitive Hashing (LSH)
- Product Quantization (PQ)

#### **Advantages**
- Computationally efficient.
- Scales well for large datasets.

---

### **7. Tools for Vector Similarity Search**
- **FAISS** (Facebook AI Similarity Search): Optimized for fast similarity search.
- **Spotify Annoy**: Efficient for approximate nearest neighbor search.

These tools were not used due to the small dataset size but are recommended for scalability.

---

## **Challenges Faced**
1. **Limited Resources:**
   - Using free Google Colab resulted in slower training times and limited GPU access.

2. **Time Constraints:**
   - Balancing this project with personal commitments (family trip) limited the implementation time to ~3 days.
  

## **Conclusion**
- **Best Performance:** ViT showed the highest retrieval accuracy but had higher computational requirements.
- **Resource-Friendly:** ResNet-50 and Siamese networks provided a good balance between performance and efficiency.
- **Future Work:** Explore FAISS or Spotify Annoy for scaling to larger datasets and implement hashing-based methods for faster search.

---

## **References**
- Pascal, Andres & Planas, Adrián & Vidal Leiva, Florencia Zoe & Bonti, Agustina & Tonelotto, Lucas & Castiglioni, León. (2024). P r e p r i n t Image Feature Extraction for Similarity Searching Using Transfer Learning with ResNet. 10.13140/RG.2.2.17101.86246.
- Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. IJCV.
- ViT Paper: “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale” by Dosovitskiy et al.
- Merugu, Suresh & Yadav, Rajesh & Pathi, Venkatesh & Perianayagam, Herbert. (2024). Identification and Improvement of Image Similarity using Autoencoder. Engineering, Technology & Applied Science Research. 14. 15541-15546. 10.48084/etasr.7548.
- Li, Yikai & Chen, C. & Zhang, Tong. (2022). A Survey on Siamese Network: Methodologies, Applications and Opportunities. IEEE Transactions on Artificial Intelligence. PP. 1-21. 10.1109/TAI.2022.3207112. 
---
