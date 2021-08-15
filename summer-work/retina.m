%{ 
Question:  
when using nvecs:
Error using eigs>checkInputs (line 265)
Number of eigenvalues requested, k, must be a positive integer not greater than n = 6. Instead, it was 7.
%}

% use the same initial matrix for all the algorithms!! 

%% Import data and visualize sample

retina1 = importdata('retina-201205_bgonlyb50RelNormconsecDel142.mat');

X = tensor(retina1);

%% 
% 1. alternating least squares for CP decomposition:

% create random initial guess
rng('default'); 
maxR = 50;
norm_X = norm(X);
nreps = 20;

errors_als_rand_init = zeros([maxR nreps]);

for r = 1:maxR
    for rep = 1:nreps
        M_als_rand_init = cp_als(X, r, 'tol',1e-6, 'maxiters',100, 'printitn', 0);

        Y = X - tensor(M_als_rand_init);
        errors_als_rand_init(r, rep) = norm(Y) / norm_X;
        
        fprintf('.');
        % show the progress and the speed
    end
    fprintf('%d (%.2f) \n',r, mean(errors_als_rand_init(r,:)));
end 

%{
% nvecs as initial guess won't work for these tensors

errors_als_nvecs_init = zeros([maxR 1]);

for r = 1:maxR
    M_als_nvecs_init = cp_als(X, min(r,6), 'init','nvecs', 'tol',1e-6, 'maxiters',100, 'printitn', 0);
    Y = X - tensor(M_als_nvecs_init);
    errors_als_nvecs_init(r) = norm(Y) / norm_X;
    
end 
%}
%%

% 2. all-at-once optimization for CP decomposition:

% without the non-negative constraint:
errors_opt = zeros([maxR nreps]);

for r = 1:maxR
    for rep = 1:nreps
        M_opt = cp_opt(X, r, 'init', 'rand', 'opt_options', struct('printEvery', 0));

        Y = X - tensor(M_opt);
        errors_opt(r, rep) = norm(Y) / norm_X;
        
        fprintf('.');
        % shwo the progress and the speed
    end
    fprintf('%d (%.2f) \n',r, mean(errors_opt(r,:)));
end 

%% with the non-negative constraint:
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

%% Plot and compare the reconstruction curves

h = figure; hold on
mean_errors_als_rand_init = zeros([maxR 1]);


for r = 1:maxR
    scatter(r * ones(nreps), errors_als_rand_init(r,:), 25, 'g', 'X');
    scatter([r], mean(errors_als_rand_init(r,:)), 25, 'g','filled');
    mean_errors_als_rand_init(r) = mean(errors_als_rand_init(r,:));
end 
a1 = line([1:maxR], mean_errors_als_rand_init, 'color', 'g');

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

%% Visualize the factors for for the best method: ALS
maxR = 20;

h = figure; hold on

F3 = zeros([maxR 8 33]);
for r = 1:maxR
    
    f3 = reshape(M_als_rand_init.u{3}(:,r),[8 33]);
    subplot(3, maxR, r);
    imagesc(f3);
    % matlab reads by columns -> need to transpose
    axis image;
    % preserve image aspect ratio
     F3(r,:,:) = f3; %arrange the first 20 factors into a matrix by rows
end
save('retina1_als_F3.mat','F3')

F1 = zeros([maxR 698]);
for r = 1:maxR
    
    f1 = M_als_rand_init.u{1}(:,r);
    subplot(3, maxR, maxR + r);
    plot(f1);
    F1(r,:) = f1; 
end
save('retina1_als_F1.mat','F1')

F2 = zeros([maxR 6]);
for r = 1:maxR
    
    f2 = M_als_rand_init.u{2}(:,r);
    subplot(3, maxR, maxR*2 + r);
    bar(f2);
    F2(r,:) = f2; 
end
save('retina1_als_F2.mat','F2')
pubgraph(h,14,2,'w')

%% Visualize the factors for for nonnegative direct optimization:

maxR = 20;

h = figure; hold on

F3 = zeros([maxR 8 33]);
for r = 1:maxR
    
    f3 = reshape(M_opt_nonneg.u{3}(:,r),[8 33]);
    subplot(3, maxR, r);
    imagesc(f3);
    axis image; % preserve image aspect ratio
    
    F3(r,:,:) = f3; %arrange the first 20 factors into a matrix by rows
end
save('retina1_opt_nonneg_F3.mat','F3')

F1 = zeros([maxR 698]);
for r = 1:maxR
    
    f1 = M_opt_nonneg.u{1}(:,r);
    subplot(3, maxR, maxR + r);
    plot(f1);
    F1(r,:) = f1; 
end
F1 =  M_opt_nonneg.u{1};
save('retina1_opt_nonneg_F1.mat','F1')

F2 = zeros([maxR 6]);
for r = 1:maxR
    
    f2 = M_opt_nonneg.u{2}(:,r);
    subplot(3, maxR, maxR*2 + r);
    bar(f2);
    F2(r,:) = f2; 
end
save('retina1_opt_nonneg_F2.mat','F2')
pubgraph(h,14,2,'w')

%% Generalized CP Decomp: Gaussian -> use the nonneg opt 

maxR = 30;
nreps = 10; 

errors_gcp = zeros([maxR nreps]);
for r = 1:maxR
    for rep = 1:nreps
        rng('default')
        [M_gcp, M0, out] = gcp_opt(X, r, 'type','normal','opt','adam','init','rand','printitn',0);
        Y = X - tensor(full(M_gcp));
        errors_gcp(r, rep) = norm(Y) / norm_X;
        fprintf('.');
    end
    fprintf('%d (%.2f) \n',r, mean(errors_gcp(r,:)));
end 

errors_gcp = errors_gcp(1:5,:);

figure; hold on
for r = 1:maxR
    scatter(r * ones(nreps), errors_gcp(r,:), 25, 'k', 'X');
    scatter([r], mean(errors_gcp(r,:)), 25, 'b','filled');
    mean_errors_gcp(r) = mean(errors_gcp(r,:));
end 
line([1:maxR], mean_errors_gcp, 'color', 'b');

%% Generalized CP Decomp: Rayleigh
maxR = 5;
nreps = 3; 

errors_gcp_rayleigh = zeros([maxR nreps]);
for r = 1:maxR
    for rep = 1:nreps
        rng('default')
        [M_gcp_rayleigh, M0, out] = gcp_opt(X, r, 'type','rayleigh', 'lower', 0,'opt','adam','init','rand','printitn',0);
        Y = X - tensor(full(M_gcp_rayleigh));
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
line([1:maxR], mean_errors_gcp_rayleigh, 'color', 'b');

%% Visualize the factors for for GCP Gaussian
maxR = 5;

h = figure; hold on

F3 = zeros([maxR 8 33]);
for r = 1:maxR
    
    f3 = reshape(M_gcp.u{3}(:,r),[8 33]);
    subplot(3, maxR, r);
    imagesc(f3);
    axis image; % preserve image aspect ratio
    
    F3(r,:,:) = f3; %arrange the first 20 factors into a matrix by rows
end

F1 = zeros([maxR 698]);
for r = 1:maxR
    
    f1 = M_gcp.u{1}(:,r);
    subplot(3, maxR, maxR + r);
    plot(f1);
    F1(r,:) = f1; 
end

F2 = zeros([maxR 6]);
for r = 1:maxR
    
    f2 = M_gcp.u{2}(:,r);
    subplot(3, maxR, maxR*2 + r);
    bar(f2);
    F2(r,:) = f2; 
end
pubgraph(h,14,2,'w')

%% Visualize the factors for for GCP Rayleigh
%% try poisson & gamma

maxR = 5;

h = figure; hold on

F3 = zeros([maxR 8 33]);
for r = 1:maxR
    
    f3 = reshape(M_gcp_rayleigh.u{3}(:,r),[8 33]);
    subplot(3, maxR, r);
    imagesc(f3);
    axis image; % preserve image aspect ratio
    
    F3(r,:,:) = f3; %arrange the first 20 factors into a matrix by rows
end

F1 = zeros([maxR 698]);
for r = 1:maxR
    
    f1 = M_gcp_rayleigh.u{1}(:,r);
    subplot(3, maxR, maxR + r);
    plot(f1);
    F1(r,:) = f1; 
end
F1 = M_gcp_rayleigh.u{1}; 

F2 = zeros([maxR 6]);
for r = 1:maxR
    
    f2 = M_gcp_rayleigh.u{2}(:,r);
    subplot(3, maxR, maxR*2 + r);
    bar(f2);
    F2(r,:) = f2; 
end
pubgraph(h,14,2,'w')

%% PCA 

[COEFF, SCORE, LATENT] = pca(F1);

%%
figure;
scatter(SCORE(:,1), SCORE(:,2));

% visualize around 20 factors 
% k-means on SCORE matrix
% plot the faces in the same cluster together

% change objective functions
% research on quantifying method 

% use tensorflow: show images and see which neurons are 







