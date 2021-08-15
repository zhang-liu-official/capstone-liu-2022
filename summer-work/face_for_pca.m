%{ 
Question:  
when using nvecs:
Error using eigs>checkInputs (line 265)
Number of eigenvalues requested, k, must be a positive integer not greater than n = 6. Instead, it was 7.
%}

% use the same initial matrix for all the algorithms!! 

%% Import data and visualize sample

% normalization method 1: 
faces_norm1 = importdata('faces_norm1.mat');
faces_norm1_X = reshape(faces_norm1,[1000 96 96]);
X = tensor(faces_norm1_X);

%% with the non-negative constraint:
rng('default'); 
maxR = 35;
norm_X = norm(X);
nreps = 20;

errors_opt_nonneg = zeros([maxR nreps]);

for r = 1:maxR
    for rep = 1:nreps

     % with the non-negative constraint:
        M_opt_nonneg = cp_opt(X, r, 'init', 'rand' , 'lower', 0, 'opt_options', struct('printEvery', 0));

        Y = X - tensor(M_opt_nonneg);
        errors_opt_nonneg(r, rep) = norm(Y) / norm_X;
        
        fprintf('.');
        % shwo the progress and the speed
    end
    fprintf('%d (%.2f) \n',r, mean(errors_opt_nonneg(r,:)));
end 

errors_opt_nonneg = errors_opt_nonneg(1:35,:);

save("errors_opt_nonneg.mat","errors_opt_nonneg");


%% Visualize the factors for nonnegative direct optimization:

h = figure; hold on

F = zeros([maxR 9216]);

for r = 1:maxR
    f = M_opt_nonneg.u{2}(:,r) * M_opt_nonneg.u{3}(:,r)';
    subplot(7, 5, r);
    imagesc(f');
    % matlab reads by columns -> need to transpose
    axis image;
    % preserve image aspect ratio
    F(r,:) = reshape(f',[1 9216]); 
end
save("factors_opt_nonneg.mat","F");
pubgraph(h,14,2,'w')

%% NEW CODE FROM HERE ON

%% PCA and kmeans for logos for testing purpose

% logos = importdata('logos.mat');
% logos = reshape(logos,[4800 10]);
% logos = logos';
% [COEFF, SCORE1, LATENT] = pca(logos','NumComponents',5);
% 
% logos_new = SCORE1(:,1)' * logos';
% logos_new = reshape(logos_new,[10 1]);
% rng(1); 
% id_logos = kmeans(logos_new,2);
%% PCA and kmeans for faces_norm1, opt_nonneg

F = importdata('factors_opt_nonneg.mat');
[COEFF, SCORE, LATENT] = pca(F','NumComponents',5);

faces_norm1_new = SCORE(:,1)' * faces_norm1';
faces_norm1_new = reshape(faces_norm1_new, [1000 1]);

save("faces_norm1_new.mat","faces_norm1_new");


% visualize around 20 factors 
% k-means on SCORE matrix
% plot the faces in the same cluster together

% change objective functions
% research on quantifying method 

% use tensorflow: show images and see which neurons are 


%% Generalized CP Decomp: Gaussian -> use the nonneg opt 

% errors_gcp = zeros([maxR nreps]);
% for r = 1:maxR
%     for rep = 1:nreps
%         rng('default')
%         [M_gcp, M0, out] = gcp_opt(X, r, 'type','normal','opt','adam','init','rand','printitn',0);
%         Y = X - tensor(full(M_gcp));
%         errors_gcp(r, rep) = norm(Y) / norm_X;
%         fprintf('.');
%     end
%     fprintf('%d (%.2f) \n',r, mean(errors_gcp(r,:)));
% end 
% 
% figure; hold on
% for r = 1:maxR
%     scatter(r * ones(nreps), errors_gcp(r,:), 25, 'k', 'X');
%     scatter([r], mean(errors_gcp(r,:)), 25, 'b','filled');
%     mean_errors_gcp(r) = mean(errors_gcp(r,:));
% end 
% line([1:maxR], mean_errors_gcp, 'color', 'b');

%% Generalized CP Decomp: Rayleigh
rng('default')
errors_gcp_rayleigh = zeros([maxR nreps]);
for r = 1:maxR
    for rep = 1:nreps
        M_gcp_rayleigh = gcp_opt(X, r, 'type','rayleigh', 'lower', 0,'init','rand','printitn',0);
        Y = X - tensor(M_gcp_rayleigh);
        errors_gcp_rayleigh(r, rep) = norm(Y) / norm_X;
        fprintf('.');
    end
    fprintf('%d (%.2f) \n',r, mean(errors_gcp_rayleigh(r,:)));
end 

figure; hold on
for r = 1:maxR
    scatter(r * ones(nreps), errors_gcp_rayleigh(r,:), 25, 'k', 'X');
    scatter([r], mean(errors_gcp_rayleigh(r,:)), 25, 'b','filled');
    mean_errors_gcp_rayleigh(r) = mean(errors_gcp_rayleigh(r,:));
    
end 
figure;
line([1:maxR], mean_errors_gcp_rayleigh, 'color', 'b');

%% Visualize the factors for GCP Rayleigh

h = figure; hold on

F = zeros([maxR 9216]);

for r = 1:maxR
    f = M_gcp_rayleigh.u{2}(:,r) * M_gcp_rayleigh.u{3}(:,r)';
    subplot(7, 5, r);
    imagesc(f');
    % matlab reads by columns -> need to transpose
    axis image;
    % preserve image aspect ratio
    F(r,:) = reshape(f',[1 9216]); 
end
save("factors_gcp_rayleigh.mat","F");
pubgraph(h,14,2,'w')

%% PCA and kmeans for faces_norm1, gcp_rayleigh

F = importdata('factors_gcp_rayleigh.mat.mat');
[COEFF, SCORE, LATENT] = pca(F','NumComponents',5);

faces_norm1_new_gcp_rayleigh = SCORE(:,1)' * faces_norm1';
faces_norm1_new_gcp_rayleigh = reshape(faces_norm1_new_gcp_rayleigh, [1000 1]);

save("faces_norm1_new_gcp_rayleigh.mat","faces_norm1_new_gcp_rayleigh");

%% Generalized CP Decomp: gamma
rng('default');
errors_gcp_gamma = zeros(maxR);
for r = 1:maxR
    %for rep = 1:nreps
        M_gcp_gamma = gcp_opt(X, r, 'type','gamma', 'lower', 0,'init','rand','printitn',0);
        Y = X - tensor(full(M_gcp_gamma));
        errors_gcp_gamma(r) = norm(Y) / norm_X;
        fprintf('.');
    %end
    %fprintf('%d (%.2f) \n',r, mean(errors_gcp_gamma(r,:)));
end 

save("errors_gcp_gamma.mat","errors_gcp_gamma");
figure;
line([1:maxR], errors_gcp_gamma, 'color', 'b');

%% Visualize the factors for GCP Gamma

h = figure; hold on

F = zeros([maxR 9216]);

for r = 1:maxR
    f = M_gcp_gamma.u{2}(:,r) * M_gcp_gamma.u{3}(:,r)';
    subplot(7, 5, r);
    imagesc(f');
    % matlab reads by columns -> need to transpose
    axis image;
    % preserve image aspect ratio
    F(r,:) = reshape(f',[1 9216]); 
end
save("factors_gcp_gamma.mat","F");
pubgraph(h,14,2,'w')

%% PCA and kmeans for faces_norm1, gcp_gamma

F = importdata('factors_gcp_gamma.mat');
[COEFF, SCORE, LATENT] = pca(F','NumComponents',5);

faces_norm1_new_gcp_gamma = SCORE(:,1)' * faces_norm1';
faces_norm1_new_gcp_gamma = reshape(faces_norm1_new_gcp_gamma, [1000 1]);

save("faces_norm1_new_gcp_gamma.mat","faces_norm1_new_gcp_gamma");

%% Plot and compare the reconstruction curves
h = figure; hold on
mean_errors_als_rand_init = zeros([maxR 1]);

for r = 1:maxR
    scatter(r * ones(nreps), errors_als_rand_init(r,:), 25, 'g', 'X');
    scatter([r], mean(errors_als_rand_init(r,:)), 25, 'g','filled');
    mean_errors_als_rand_init(r) = mean(errors_als_rand_init(r,:));
end 
a1 = line([1:maxR], mean_errors_opt_nonneg, 'color', 'g');

%a2 = plot(errors_als_nvecs_init, 'color', 'g');

mean_errors_opt = zeros([maxR 1]);

for r = 1:maxR
    scatter(r * ones(nreps), errors_opt(r,:), 25, 'm', 'X');
    scatter([r], mean(errors_opt(r,:)), 25, 'm','filled');
    mean_errors_opt(r) = mean(errors_opt(r,:));
end 
a3 = line([1:maxR], mean_errors_opt, 'color', 'm');

mean_errors_opt_nonneg = zeros([maxR 1]);

for r = 1:maxR
    scatter(r * ones(nreps), errors_opt_nonneg(r,:), 25, 'b', 'X');
    scatter([r], mean(errors_opt_nonneg(r,:)), 25, 'b','filled');
    mean_errors_opt_nonneg(r) = mean(errors_opt_nonneg(r,:));
end 
a4 = line([1:maxR], mean_errors_opt_nonneg, 'color', 'b');

legend([a1;a3;a4], 'ALS with random initial guess', 'Regular Direct Optimization', 'Nonnegative Direct Optimization');

title("Reconstruction error curves (Retina 1)");

%syntax: pubgraph(FigureHandle, FontSize, LineWidth, Color)
pubgraph(h,20,5,'w')
