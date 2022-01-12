%% Import data

neuron_output = importdata('neuron_output_2D_epoch10.mat');

X = tensor(neuron_output);

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
        % show the progress and the speed
    end
    fprintf('%d (%.2f) \n',r, mean(errors_opt_nonneg(r,:)));
end 
F = M_opt_nonneg.u{1};
save("factors_opt_nonneg_2D_epoch10.mat","F");

%% Plot error curve:
mean_errors_opt_nonneg = zeros([maxR 1]);

for r = 1:maxR
    scatter(r * ones(nreps), errors_opt_nonneg(r,:), 25, 'b', 'X');
    scatter([r], mean(errors_opt_nonneg(r,:)), 25, 'b','filled');
    mean_errors_opt_nonneg(r) = mean(errors_opt_nonneg(r,:));
end 
a4 = line([1:maxR], mean_errors_opt_nonneg, 'color', 'b');
legend(a4, 'Nonnegative Direct Optimization');

title("Reconstruction error curves (VGG-16 Block 1)");

%syntax: pubgraph(FigureHandle, FontSize, LineWidth, Color)
pubgraph(h,20,5,'w')
M_opt_nonneg.u{3}(:,1).shape
%% Visualize the factors for for nonnegative direct optimization:

maxR = 20;

h = figure; hold on

% neuron tensor factor (10240, 20)

F1 = zeros([maxR 1210]);
for r = 1:maxR
    
    f1 = M_opt_nonneg.u{1}(:,r);
    subplot(2, maxR, r);
    plot(f1);
    F1(r,:) = f1; 
end
F1 =  M_opt_nonneg.u{1};
%save('opt_nonneg_F1.mat','F1')

% neuron tensor factor (10240, 20)
F2 = zeros([maxR 40]);
for r = 1:maxR
    
    f2 = M_opt_nonneg.u{2}(:,r);
    subplot(2, maxR, maxR + r);
    bar(f2);
    F2(r,:) = f2; 
end
%save('opt_nonneg_F2.mat','F2')
pubgraph(h,14,2,'w')

%% Generalized CP Decomp: Rayleigh
maxR = 35;
%nreps = 1; 

errors_gcp_rayleigh = zeros(maxR);
for r = 1:maxR
    %for rep = 1:nreps
        rng('default')
        [M_gcp_rayleigh, M0, out] = gcp_opt(X, r, 'type','rayleigh','opt','adam', 'lower', 0, 'init','rand','printitn',0);
        Y = X - tensor(full(M_gcp_rayleigh));
        errors_gcp_rayleigh(r) = norm(Y) / norm_X;
        fprintf('.');
    %end
    fprintf('%d (%.2f) \n',r, errors_gcp_rayleigh(r));
end 
F = M_gcp_rayleigh.u{1};
save("factors_gcp_rayleigh.mat","F");

figure; hold on
for r = 1:maxR
    scatter(r * ones(nreps), errors_gcp_rayleigh(r,:), 25, 'k', 'X');
    scatter([r], mean(errors_gcp_rayleigh(r,:)), 25, 'b','filled');
    mean_errors_gcp_rayleigh(r) = mean(errors_gcp_rayleigh(r,:));
end 
line([1:maxR], mean_errors_gcp_rayleigh, 'color', 'b');

%% Visualize the factors for for GCP Rayleigh
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