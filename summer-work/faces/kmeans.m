function [clusters centroids] = kmeans(X, k, init)
%KMEANS implements the k-means algorithm.
%   [CLUSTERS CENTROIDS] = KMEANS(X, K, INITS) partitions the data points
%   in the N-by-P data matrix X into K distinct clusters, using Euclidean
%   distance. This is a simple implementation of the k-means algorithm with
%   random initialization.
%
%   Optionally, it takes the argument INIT, a K-by-P matrix with a fixed
%   initial position for the cluster centroids.  
%
%   MYKMEANS returns an N-by-1 vector CLUSTERS containing the cluster
%   indices of each data point, as well as CENTROIDS, a K-by-P matrix with 
%   the final cluster centroids' locations.


[n p] = size(X);

if ~exist('init', 'var')
  %choose initial centroids by picking k points at random from X
  init = X(randi(n, 1, k), :);
end
%centroids is a k-by-p random matrix
%its i^th row contains the coordinates of the cluster with index i
centroids = init;

%initialize cluster assignment vector
clusters = zeros(1,n);

MAXITER = 1000;

for iter=1:MAXITER
    
    %create a new clusters vector to fill in with updated assignments
    new_clusters = zeros(1,n);
    

    %for each data point x_i
    for i=1:n
        
        x_i = X(i,:);
        
        %find closest cluster
        closest = findClosestCluster(x_i,centroids);%%%IMPLEMENT THIS FUNCTION AT THE END OF THIS FILE

        %reassign x_i to the index of the closest centroid found
        new_clusters(i) = closest;

    end
    
    if hasConverged(clusters,new_clusters)%%%IMPLEMENT THIS FUNCTION AT THE END OF THIS FILE
        %exit loop
        break 
    end
    
    %otherwise, update assignment
    clusters = new_clusters;
    %and recompute centroids
    centroids = recomputeCentroids(X,clusters,k);%%%IMPLEMENT THIS FUNCTION AT THE END OF THIS FILE

end

if iter == MAXITER
    disp('Maximum number of iterations reached!');
end

end

function closest = findClosestCluster(x_i,centroids)
% Compute Euclidean distance from x_i to each cluster centroid and return 
% the index of the closest one (an integer).
% NOTE: use of MATLAB's `norm` function is NOT allowed here.

%%% Replace the following line with your own code
closest = 1;

end

function converged = hasConverged(old_assignment, new_assignment)
% Check if algorithm has converged, i.e., cluster assignments haven't
% changed since last iteration. Return a boolean.

%%% Replace the following line with your own code
converged = true;

end

function centroids = recomputeCentroids(X,clusters,k)
% Recompute centroids based on current cluster assignment.
% Return a k-by-p array where each row is a centroid. 
[n p] = size(X);
%%% Replace the following line with your own code
centroids = zeros(k,size(X,2));

end
    