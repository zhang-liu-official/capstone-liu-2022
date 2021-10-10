# Capstone Liu 2022

**Manifold Structure of High-Dimensional Data in Visual Perception**

*Key words: manifold learning, recurrent neural networks, computational vision, neural data analysis, harmonic analysis*

## Proposal:

This capstone project builds on a long line of research aiming to develop more accurate mathematical and computational models of early vision in the primary visual cortex (V1). It has recently been shown that the common feed-forward neural networks popularized by the success of deep learning turn out to be far from the neurobiological networks in V1. However, a question that has not been investigated is how the structure of recurrent and transformer neural networks relates to that of neurobiological networks. A preliminary hypothesis is that the structure of the recurrent and transformer neural networks are not accurate models of the visual system either, which we would like to verify in this project. 

To this end, the main approach of this project is to infer manifolds of neurons (either biological or artificial) in which neurons that respond similarly to an ensemble of stimuli are defined to be closer to each other. This manifold structure implies a functional network (the discrete data graph underlying the continuous manifold) and thus reflects both the neural circuit connections and the neuron’s role in those circuits. One of the key challenges in discovering the underlying manifold structure is the high-dimensionality of the neural data. It is thus necessary to apply both linear and non-linear dimensionality reduction methods, including tensor factorization and diffusion maps. 

The significance of this project is threefold. A first and more immediate contribution would be novel findings regarding whether recurrent models are accurate representations of V1. In a broader view, successful completion of this project can push the frontier of the field with a step closer towards better models of early vision, potentially offering insights for reverse-engineering new computer vision algorithms. Last but not least, the techniques used in this project are applicable to any high-dimensional data, which are prevalent in numerous areas beyond the sciences.


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


Resources:

1. On tensor factorization:
1) Matrix/tensor factorization, by the first author of the paper you've read:
https://www.youtube.com/watch?v=hmmnRF66hOA [Part 1 - matrices]
https://www.youtube.com/watch?v=O-YTsSuEFiM [Part 2 - tensors]

2) A talk on generalized CP decomp. and applications, by Tamara Kolda:
https://www.youtube.com/watch?v=L8uT6hgMt00

3) This one is more math-oriented; feel free to focus only on the first 18 min:
https://www.youtube.com/watch?v=HcIN27_WqPU
