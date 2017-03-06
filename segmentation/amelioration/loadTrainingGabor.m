% ******** This file is an update of the "LoadAppleScript" furnished with
% the homework **
function r = loadTrainingGabor

% Note that cells are accessed using curly-brackets {} instead of parentheses ().
Iapples = cell(3,1);
Iapples{1} = '../../trainingApples/Apples_by_kightp_Pat_Knight_flickr.jpg';
Iapples{2} = '../../trainingApples/ApplesAndPears_by_srqpix_ClydeRobinson.jpg';
Iapples{3} = '../../trainingApples/bobbing-for-apples.jpg';

IapplesMasks = cell(3,1);
IapplesMasks{1} = '../../trainingApples/Apples_by_kightp_Pat_Knight_flickr.png';
IapplesMasks{2} = '../../trainingApples/ApplesAndPears_by_srqpix_ClydeRobinson.png';
IapplesMasks{3} = '../../trainingApples/bobbing-for-apples.png';

% Save a result as an example 
A = gaborFilter(rgb2gray(double(imread(Iapples{1})) / 255), 30,  15, 0.5) ;
imwrite(A,'gaborFilter.jpg');
imshow(A) ;

for i=1:3
    iImage = i;
    curI = double(imread(  Iapples{iImage}   )) / 255;
    
    % cell and initialization for the gabor results
    IgaborResult = cell(9,1) ;
    rotationCount = 0 ;
    sigmaCount = 1 ;
    
    % creation of gabor filter results in a new cell.
    for theta= 1:59:119
        for sigm = 10:5:20
            IgaborResult{rotationCount+sigmaCount} = gaborFilter(rgb2gray(curI),theta, sigm, 0.5) ;
            sigmaCount=sigmaCount+1 ;
        end
        sigmaCount = 1 ;
        rotationCount=rotationCount+3 ;
    end
    
    [M,N,~] = size(curI) ; % the size of the current image
    curImask = imread(IapplesMasks{iImage});
    % These mask-images are often 3-channel, and contain grayscale values. We
    % would prefer 1-channel and just binary:
    curImask = curImask(:,:,2) > 128;  % Picked green-channel arbitrarily.
    
    A = sum(curImask(:) > 0) ; % total number of apple pixels on the image
    
    % We define the 2 matrixs containing the pixel data for apple image and
    % non apple image and the size is now 12, with the RGB values and the
    % gabor filter.
    RGBApple = zeros(3+size(IgaborResult,1),A) ;
    RGBNonApple = zeros(3+size(IgaborResult,1), M*N-A) ;
    countNonApple= 1;
    countApple = 1;
   
    for m = 1:M
        for n=1:N
            if(curImask(m,n)) % if the pixel is a apple pixel, we add it to the matrix
                RGBApple(1:3,countApple) = reshape(curI(m,n,:),3,1) ;
                % we add all the result from the gabor filtering to our
                % matrix
                for gaborIndex = 1:rotationCount*sigmaCount
                    RGBApple(3+gaborIndex,countApple) = IgaborResult{gaborIndex}(m,n)  ;
                end
                countApple = countApple+1 ;
                
            else % else we add it to the other matrix
                RGBNonApple(1:3,countNonApple) = reshape(curI(m,n,:),3,1) ;
                % we add all the result from the gabor filtering
                for gaborIndex = 1:size(IgaborResult,1)
                    RGBNonApple(3+gaborIndex,countNonApple) = IgaborResult{gaborIndex}(m,n)  ;
                end
                
                countNonApple = countNonApple+1 ;
            end
        end
    end
    
    if (i==1) % we create the final matrix with the first image data
        RGBNonAppleFinal = RGBNonApple ;
        RGBAppleFinal = RGBApple ;
        priorApple = A/(M*N) ;
    else % we add the other data image by concatenation
        RGBNonAppleFinal = cat(2, RGBNonAppleFinal, RGBNonApple) ;
        RGBAppleFinal = cat(2,RGBAppleFinal, RGBApple);
        priorApple = (priorApple + A/(M*N))/2 ; % we compute the average prior of a pixel to be an apple
    end
    
end
% we save our data
save('RGBTrainedGabor.mat','priorApple', 'RGBAppleFinal','RGBNonAppleFinal') ;

end

% function to perform the filtering in frenquency domain
function result = gaborFilter(img, angle,  sigma, lambda)

% img is the original image
% angle is the angle of the gobor filter
% sigma is the variance of the gaussian envelop
% lambda is the wavelength of the sinusoidal factor

[rows, cols] = size(img);
windowS = 15 ; %size of the filter window

% we create a meshgrid to use our filter
[X, Y] = meshgrid(-floor(windowS/2):floor(windowS/2) , -floor(windowS/2):floor(windowS/2)) ;

% we compute the rotation
X1 = X.*cos(angle)+Y.*sin(angle) ;
Y1 = -X.*sin(angle)+Y.*cos(angle) ;

% creation of the two part of the filter, the gaussian and the rotation
% part
gauss_part = 1/(2*pi*sigma^2)*exp(-((X1.^2+Y1.^2)./(2*sigma^2))) ;
rot_part = exp(-2*pi*1i*X1/lambda) ;

% put the image and the filter in the frequency domain with good dimension
gabor_fft = fft2(gauss_part*rot_part,rows,cols);
img_fft = fft2(img, size(gabor_fft, 1), size(gabor_fft, 2));

% Perform filtering.
result = real(ifft2(gabor_fft.*img_fft));
% Crop to original size.
result = result(1:size(img, 1), 1:size(img, 2));

% rescale the result image between 0 and 1
result = (result - min(result(:))) / (max(result(:)) - min(result(:))) ;

end