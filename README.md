# L2T-Hyena: Enhancing State-Space Models with an Adaptive Learn-to-Teach Framework
State-Space Models (SSMs) have emerged as efficient alternatives to computationally intensive architectures like Transformers, particularly for sequence modeling. However, a fundamental challenge in their training is the reliance on static loss functions, which may not be optimal across all learning stages. To address this issue, in this paper a hybrid model integrating the Hyena architecture with a Dynamic Loss Network (DLN) is proposed which is guided by a Learn-to-Teach (L2T) approach (L2T-DLN). In this framework, the Hyena model is a student, and its loss function is optimized adaptively. A teacher model, leveraging a memory of the student's past performance, guides the DLN in dynamically balancing the primary cross-entropy loss and a regularization term. Experiments on the Penn Treebank (PTB) dataset show that our approach significantly improves language modeling performance. Our proposed model achieved a validation Perplexity of 102.6, a notable improvement over the 110.4 achieved by a baseline Hyena model using a static loss function. This research indicates that combining SSMs with adaptive loss function markedly enhances the quality and efficiency of deep learning models for sequential data, showing potential for applications in Natural Language Processing (NLP), time-series analysis, and biological signal processing.

Model Architecture
Our proposed architecture, L2T-Hyena, integrates the Hyena student model within the L2T-DLN framework. This system consists of three main components:
1.	Student Model: A Hyena-based architecture for language modeling.
2.	Dynamic Loss Network (DLN): Dynamically adjusts the weights between the Cross-Entropy loss and an L2 logit regularization term.
3.	Teacher Model: A memory-augmented system that guides the DLN's learning process by predicting the student's final loss.
The overall structure and information flow of the L2T-DLN framework are shown in the figure below:
<img width="504" height="310" alt="image" src="https://github.com/user-attachments/assets/06a1478b-45b6-4b91-ada2-ee4f6cf6caed" />
Results
We evaluated our model on the Penn Treebank (PTB) dataset. The results show that L2T-Hyena significantly improves performance compared to the baseline Hyena model trained with a static loss function.
•	L2T-Hyena (Our Model): Validation Perplexity = 102.6
•	Baseline Hyena: Validation Perplexity = 110.4
This 7.1% improvement confirms the effectiveness of our adaptive training framework.
<img width="455" height="344" alt="image" src="https://github.com/user-attachments/assets/19afea0f-8b92-4d49-a4b5-6eb448ef1554" />
The chart shows that the baseline Hyena model (blue) quickly begins to overfit at epoch 3, whereas the L2T-Hyena model (red) achieves a better perplexity and maintains its stability.
