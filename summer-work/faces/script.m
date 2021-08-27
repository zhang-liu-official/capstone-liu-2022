data_tinypics = importdata('tinypics.mat');
imagesc(x);

data_logos = importdata('logos.mat');

t = tiledlayout(2,5);
X = zeros(10,4800);
dim = zeros(10);

for i = 1:10
    image_i = data_logos(:,:,i);
    X(i,:) = image_i(:);
    dim = size(X(i,:));
    nexttile;
    imagesc(image_i);
end

%gplotmatrix(X,[],['c' 'b' 'm' 'g' 'r'],[],[],false);

