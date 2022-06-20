# Capstone Liu 2022

**Manifold Structure of Artificial and Biological Neural Networks**

*Key words: manifold learning, neural computation, early primate vision, computational vision*

## Abstract:
Our work builds on a long line of research aiming to develop more accurate computational models of the visual system. Despite decades of research, we have not fully understood the structure of neural circuits responsible for visual perception. Additionally, among the computer vision (CV) community, there is a growing interest in investigating the limitations of the CV models by comparing them against biological vision. To this end, we seek to answer two specific open questions: First, how do CV models (VGG16, Vision Transformer, and Convolutional Recurrent Neural Network) compare to biological vision, at retina and primary visual cortex (V1), in terms of their respective neural circuits1? Second, what specific mechanisms are important in causing such differences and/or similarities?

We first build the biological neural tensors using experimental neural spiking data and artificial neural tensors using numerical simulations on computer vision models (VGG16, Vision Transformer, and Convolutional Recurrent Neural Network). Using tensor CP decomposition, we obtain a manifold of neurons. The discrete data graph underlying the manifold then reflects both the neural circuit connections and the neurons’ role in those circuits. By comparing the manifold structure of neurobiological networks in retina and V1 with that of computer vision models, we can make precise inferences about similarities and differences in their respective functional circuits. For the first time, we find that the underlying neural circuit of feed-forward computer vision models including CNN and ViT form disconnected network clusters, making them poor approximations of the visual cortex, contrary to popular belief. In order to model the highly connected neural circuits in the visual cortex, recurrent structure is likely necessary.


## Milestones:

Milestones for semester 1:

(Theory)

1. Study CPSC475 course materials shared by the co-supervisor Prof. Zucker
2. Read papers related to manifold learning, recurrent and transformer neural networks, harmonic analysis
3. Study the theory of Diffusion Maps papers in details
4. Write up literature review and notes for important concepts

(Implementation)

5. Implement Diffusion Maps for the homework problems
6. Implement Diffusion Maps for the neural spiking data with biological and artificial neuron tensors
7. Review Diffusion Maps implementation with co-supervisor
8. Implement artificial neuron tensors
9. Implement RNN with LSTM and Transformer NN
10. Finish report 1 


Milestones for semester 2: (to be reviewed and planned in greater details after the end of semester 1)

1. Finish the rest of the implementations
2. Testing the hypothesis whether the manifold structure of recurrent and transformer neural networks are similar to that in neurobiological neural networks
3. Analyze and evaluate the results
4. Depending on the results, we will decide on the next steps which could be 
    (1) conduct further experiments to test for other aspects that we have neglected/overlooked or modify some parts of the experimental procedures; 
    (2) propose a more accurate model for visual processing in V1
    (3) study the theory in greater depth to improve the current methodology
5. Finish report
6. Finish presentation
