% ******** This file is an update of the "LoadAppleScript" furnished with
% the homework **

% Note that cells are accessed using curly-brackets {} instead of parentheses ().
im1 = imread('../../trainingApples/Apples_by_kightp_Pat_Knight_flickr.jpg') ;
im2 = imread('../../trainingApples/ApplesAndPears_by_srqpix_ClydeRobinson.jpg') ;
im3 = imread('../../trainingApples/bobbing-for-apples.jpg') ;

% Here I use rgb2hsv to convert the color space
Iapples = cell(3,1);
Iapples{1} = rgb2hsv(im1);
Iapples{2} = rgb2hsv(im2);
Iapples{3} = rgb2hsv(im3);
% after that the code is similar 

IapplesMasks = cell(3,1);
IapplesMasks{1} = '../../trainingApples/Apples_by_kightp_Pat_Knight_flickr.png';
IapplesMasks{2} = '../../trainingApples/ApplesAndPears_by_srqpix_ClydeRobinson.png';
IapplesMasks{3} = '../../trainingApples/bobbing-for-apples.png';

% For each image, I create a 3*nData matrix of apple pixels and an other one with non apple
% pixels
for i=1:3

    curI = Iapples{i} ;
    
    [M,N,~] = size(curI) ;
    curImask = imread(  IapplesMasks{i}   );
    curImask = curImask(:,:,2)> 128;  % Picked green-channel arbitrarily.
   
    A = sum(curImask(:) > 0) ; % total number of apple pixels on the image
    
    % We define the 2 matrixs containing the pixel data for apple image and
    % non apple image 
    RGBApple = zeros(3,A) ;
    RGBNonApple = zeros(3, M*N-A) ;
    
    countNonApple= 1; 
    countApple = 1; 
    for m = 1:M
        for n=1:N
            if(curImask(m,n)) % if the pixel is a apple pixel, we add it to the matrix
                RGBApple(:,countApple) = reshape(curI(m,n,:),3,1) ;
                countApple = countApple+1 ;
            else % else we add it to the other matrix
                RGBNonApple(:,countNonApple) = reshape(curI(m,n,:),3,1) ;
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
save('RGBTrainedHSV.mat','priorApple', 'RGBAppleFinal','RGBNonAppleFinal') ;
