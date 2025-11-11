# L2T-Hyena: Enhancing State-Space Models with an Adaptive Learn-to-Teach Framework
State-Space Models (SSMs) have emerged as efficient alternatives to computationally intensive architectures like Transformers, particularly for sequence modeling. However, a fundamental challenge in their training is the reliance on static loss functions, which may not be optimal across all learning stages. To address this issue, in this paper a hybrid model integrating the Hyena architecture with a Dynamic Loss Network (DLN) is proposed which is guided by a Learn-to-Teach (L2T) approach (L2T-DLN). In this framework, the Hyena model is a student, and its loss function is optimized adaptively. A teacher model, leveraging a memory of the student's past performance, guides the DLN in dynamically balancing the primary cross-entropy loss and a regularization term. Experiments on the Penn Treebank (PTB) dataset show that our approach significantly improves language modeling performance. Our proposed model achieved a validation Perplexity of 102.6, a notable improvement over the 110.4 achieved by a baseline Hyena model using a static loss function. This research indicates that combining SSMs with adaptive loss function markedly enhances the quality and efficiency of deep learning models for sequential data, showing potential for applications in Natural Language Processing (NLP), time-series analysis, and biological signal processing.

Model Architecture
Our proposed architecture, L2T-Hyena, integrates the Hyena student model within the L2T-DLN framework. This system consists of three main components:
Student Model: A Hyena-based architecture for language modeling.
Dynamic Loss Network (DLN): Dynamically adjusts the weights between the Cross-Entropy loss and an L2 logit regularization term.
Teacher Model: A memory-augmented system that guides the DLN's learning process by predicting the student's final loss.
The overall structure and information flow of the L2T-DLN framework are shown in the figure below:

<img width="504" height="310" alt="image" src="https://github.com/user-attachments/assets/06a1478b-45b6-4b91-ada2-ee4f6cf6caed" />

Fig 1: The pipeline of L2T-DLN 

Experimental Setup
We evaluated our model on the Penn Treebank (PTB) dataset. The system was implemented using PyTorch, and each of the three components (Student, Teacher, DLN) was trained using the AdamW optimizer with distinct hyperparameters, as detailed in Table 1.

<img width="459" height="283" alt="image" src="https://github.com/user-attachments/assets/afd8181b-4f93-4f69-a398-992935ef5dea" />

Table 1: Optimizer Hyperparameters

Results and Discussion
We compared the performance of the L2T-Hyena model against a baseline Hyena (Vanilla Hyena) model trained with a static loss function.
Quantitative Performance Comparison
Table 2 summarizes the final numerical results and comparison between the two models.

<img width="485" height="324" alt="image" src="https://github.com/user-attachments/assets/1114b998-af68-4abc-a61b-05bec137a088" />

Table 2: Detailed Performance Comparison on PTB Dataset

As shown, the L2T-Hyena model achieved a final validation perplexity of 102.6, which is a 7.1% improvement over the baseline's score of 110.4. This result confirms the effectiveness of our adaptive training framework.
Stability and Overfitting
Figure 2 illustrates the validation perplexity over the 10 training epochs.

<img width="455" height="344" alt="image" src="https://github.com/user-attachments/assets/fe622d2d-86b4-4909-b74a-0c395959efcf" />

Fig 2: Comparison of Validation Perplexity

The chart clearly shows that the baseline Hyena model (blue) quickly begins to overfit after epoch 3. In contrast, the L2T-Hyena model (red) not only achieves a better perplexity but also maintains significant stability in later epochs, demonstrating the L2T-DLN framework's ability to mitigate overfitting and improve generalization.
Training Loss Dynamics
Figure 3 compares the average training loss for both models.

<img width="463" height="331" alt="image" src="https://github.com/user-attachments/assets/19cc986c-82d1-48fb-a99c-3056bd4a0269" />

Fig 3: Comparison of Training Loss

This plot indicates that the proposed L2T-Hyena model converges significantly faster and to a lower final loss value than the baseline. This suggests a more efficient learning process on the training data, which also translated to better generalization (as seen in Figure 2).



Citation
@misc{sohbati2025l2thyena,
  title={{L2T-Hyena: Enhancing State-Space Models with an Adaptive Learn-to-Teach Framework}},
  author={Sohbati, Fatemeh and Haddadi, Farzan and Salahinejad, Hamid},
  year={2025},
  publisher={arXiv},
  eprint={2511.05926},
  archivePrefix={arXiv},
  primaryClass={cs.IT}
}
