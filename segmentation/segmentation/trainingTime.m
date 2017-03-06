% ******** This file is an update of the "LoadAppleScript" furnished with
% the homework **

% Note that cells are accessed using curly-brackets {} instead of parentheses ().
Iapples = cell(3,1);
Iapples{1} = '../../trainingApples/Apples_by_kightp_Pat_Knight_flickr.jpg';
Iapples{2} = '../../trainingApples/ApplesAndPears_by_srqpix_ClydeRobinson.jpg';
Iapples{3} = '../../trainingApples/bobbing-for-apples.jpg';

IapplesMasks = cell(3,1);
IapplesMasks{1} = '../../trainingApples/Apples_by_kightp_Pat_Knight_flickr.png';
IapplesMasks{2} = '../../trainingApples/ApplesAndPears_by_srqpix_ClydeRobinson.png';
IapplesMasks{3} = '../../trainingApples/bobbing-for-apples.png';

% For each image, I create a 3*nData matrix of apple pixels and an other one with non apple
% pixels

for i=1:3
    iImage = i;
    curI = double(imread(  Iapples{iImage}   )) / 255;
    % curI is now a double-precision 3D matrix of size (width x height x 3).
    % Each of the 3 color channels is now in the range [0.0, 1.0].
    
    [M,N,~] = size(curI) ; % the size of the current image
    curImask = imread(  IapplesMasks{iImage}   );
    % These mask-images are often 3-channel, and contain grayscale values. We
    % would prefer 1-channel and just binary:
    curImask = curImask(:,:,2) > 128;  % Picked green-channel arbitrarily.
   
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
save('RGBTrained.mat','priorApple', 'RGBAppleFinal','RGBNonAppleFinal') ;

