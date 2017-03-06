function r = QuestionD 

% we are going to plot the ROC curve of the results for the same image using different number
% of gaussian to find which number is the best one

% we take the original image and the groundTruth to compute the ROC curve
groundTruth = imread('../../validationApples/groundTruth.png') ;
groundTruth = rgb2gray(groundTruth) ;

% We compute the ROC curve for each posterior results according to the number of Gaussian. 
%The posterior result have been computed at the end of QuestionC 
postROC3 = imread('../../validationApples/postROC3.jpg') ;
[X1, Y1] = computeROCCurve(groundTruth, postROC3) ;

postROC5 = imread('../../validationApples/postROC5.jpg') ;
[X2, Y2] = computeROCCurve(groundTruth, postROC5) ;

postROC7 = imread('../../validationApples/postROC7.jpg') ;
[X3, Y3] = computeROCCurve(groundTruth, postROC7) ;

postROC10 = imread('../../validationApples/postROC10.jpg') ;
[X4, Y4] = computeROCCurve(groundTruth, postROC10) ;

% For each curve, we find the distance to (0,1) which will be the best
% result possible with this number of gaussian according to the groundTruth of the image 
[dist1, thres1] = findThreshold(X1,Y1) ;
[dist2, thres2] = findThreshold(X2,Y2) ;
[dist3, thres3] = findThreshold(X3,Y3) ;
[dist4, thres4] = findThreshold(X4,Y4) ;

fig1 = figure ;
set(fig1, 'Position', [0 0 1000 400]),
subplot(2,4,1),
imshow(postROC3,[min(postROC3(:)) max(postROC3(:))])
subplot(2,4,5),
plot(X1,Y1, 'r-','LineWidth',1.2),
title(['ROC with 3 gaussians. Dist = ' num2str(dist1) ]) ; 

subplot(2,4,2),
imshow(postROC5,[min(postROC5(:)) max(postROC5(:))])
subplot(2,4,6),
plot(X2,Y2, 'r-','LineWidth',1.2),
title(['ROC with 5 gaussians. Dist = ' num2str(dist2) ]) ;

subplot(2,4,3),
imshow(postROC7,[min(postROC7(:)) max(postROC7(:))])
subplot(2,4,7),
plot(X3,Y3, 'r-','LineWidth',1.2),
title(['ROC with 7 gaussians. Dist = ' num2str(dist3) ]) ;

subplot(2,4,4),
imshow(postROC10,[min(postROC10(:)) max(postROC10(:))])
subplot(2,4,8),
plot(X4,Y4, 'r-','LineWidth',1.2),
title(['ROC with 10 gaussians. Dist = ' num2str(dist4) ]) ;

saveas(gcf, 'questionD', 'jpg'); % save the result

end

% function to threshold our posterior result. If a pixel intensity is above
% the threshold, it's intensity is now 255, if it's bellow, its intensity
% is now 0. 
function B = threshold(img, k)
% img is the original image we want to segment
% k is the threshold we choose
% B is the binary result of the segmentation 
[M, N] = size(img) ;
B = zeros(M,N, 'uint8') ;
for i=1:M
    for j=1:N
        if(img(i,j) >= k) % if the pixel intensity value is above the threshold, we set his value to 255
            B(i,j) = 255 ;
        else % if not, the pixel will be black
            B(i,j) = 0;
        end
    end
end
B = uint8(B) ;
end

% using that function we can know which number of gaussian produces the
% best result 
function [distValue thresValue] = findThreshold(X, Y)
sizeX = size(X,2) ;
A = ones(1,sizeX) ;
dist = sqrt(X.^2+(Y-A).^2);
distValue = min(dist) ;
minDiff = find(dist==min(dist)) ;
thresValue = minDiff(1) ;
end


% this routine allows to compute the ROC curve for a given posterior and a
% groundtruth 
function [X, Y] = computeROCCurve(groundTruth, posteriorApple)
[M, N] = size(posteriorApple) ;
tp = 0; % number of true positive
tn = 0 ; % number of true negative
fp = 0 ; % number of false positive
fn = 0 ; % number of false negative

numberOfThreshold = 255 ; %the number of threshold we will use to plot the ROC curve
X = zeros(1,numberOfThreshold) ; % contain the false positive rate
Y = zeros(1,numberOfThreshold) ; % contain the true positive rate

for i= 1:numberOfThreshold
    A = threshold(posteriorApple,i) ;
    for m=1:M
        for n=1:N
            if(groundTruth(m,n) == 0 && A(m,n) ==0)
                tn = tn+1 ;
            elseif(groundTruth(m,n) == 0 && A(m,n) == 255)
                fp = fp + 1 ;
            elseif(groundTruth(m,n) == 255 && A(m,n) == 0)
                fn = fn +1 ;
            elseif(groundTruth(m,n) == 255 && A(m,n) == 255)
                tp = tp +1 ;
            end
        end
    end
    X(1, i) = fp/(tn+fp) ; % false positive rate
    Y(1,i) = tp/(tp+fn) ; %true positive rate
    fp = 0;
    fn = 0 ;
    tp =0 ;
    tn = 0 ;
end
end

