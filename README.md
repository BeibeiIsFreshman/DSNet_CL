# Turbidity-Similarity Decoupling: Feature-Consistent Mutual Learning for Underwater Salient Object Detection
# Data
Please download the USOD10K dataset and training/testing framework from https://github.com/LinHong-HIT/USOD10K.
The USOD dataset is a new RGB-D dataset that we generated using the depth estimation algorithm recommended by USOD10K. Please download this dataset from the model weights download link that will be provided later.

# Essay
Abstract—Underwater salient object detection (USOD) faces two major challenges—substantial image noise due to water turbidity and low foreground-background contrast caused by visual similarity—which hinder accurate detection. To address these issues, a dual-model architecture based on mutual learning is proposed. First, DenoisedNet, which focuses on addressing water turbidity issues, is developed using a separation-denoising-enhancement processing framework that suppresses noise while maintaining target feature integrity through the domain separation and cleaning enhancement modules. Second, SearchNet is designed to address the foreground-background similarity issue, achieving precise localization through pseudo-label generation and layer-by-layer search mechanisms. To enable both networks to collaboratively address these challenges, a mutual-learning strategy based on feature consistency is proposed, incorporating evaluation and cross modes for encoded features and prediction results, respectively. This strategy promotes mutual learning between the two models through feature alignment, allowing their respective strengths to be complemented and the challenges of USOD to be solved more comprehensively. The results of experiments on the USOD10K and USOD benchmark datasets demonstrate that the proposed method outperforms existing approaches across all evaluation metrics, successfully overcoming noise and detection challenges.

The diagram of our model is as follows:
<img width="753" height="716" alt="image" src="https://github.com/user-attachments/assets/06b455c3-998a-44f5-ba1f-803e3bacab7b" />

The results of our comparison method are as follows:
<img width="847" height="669" alt="image" src="https://github.com/user-attachments/assets/fffb92ca-d461-4230-9e74-39064a89ce8d" />

# Weight
The backbone network adopts PVTv2 and MixTransformer. Please visit its official GitHub repository to download the network code and pre-trained weights.

# environment
Refer to requirements.txt for the environment configuration file.

