function HW2_TrackingAndHomographies


LLs = HW2_Practical9d( 'll.mat' );
LRs = HW2_Practical9d( 'lr.mat' );
ULs = HW2_Practical9d( 'ul.mat' );
URs = HW2_Practical9d( 'ur.mat' );

close all;

LoadVideoFrames

% Coordinates of the known target object (a dark square on a plane) in 3D:
XCart = [-50 -50  50  50;...
    50 -50 -50  50;...
    0   0   0   0];

% These are some approximate intrinsics for this footage.
K = [640  0    320;...
    0    512  256;
    0    0    1];

% Define 3D points of wireframe object.
XWireFrameCart = [-50 -50  50  50 -50 -50  50  50;...
    50 -50 -50  50  50 -50 -50  50;...
    0   0   0   0 -100 -100 -100 -100];

hImg = figure;

for iFrame = 1:numFrames
    xImCart = [LLs(iFrame,:)' ULs(iFrame,:)' URs(iFrame,:)' LRs(iFrame,:)'];
    xImCart = circshift( xImCart, 1);
    
    % To get a frame from footage
    im = Imgs{iFrame};
    
    % Draw image and 2d points
    set(0,'CurrentFigure',hImg);
    set(gcf,'Color',[1 1 1]);
    imshow(im); axis off; axis image; hold on;
    plot(xImCart(1,:),xImCart(2,:),'r.','MarkerSize',15);

    T = estimatePlanePose(xImCart, XCart, K);
    
    %Draw a wire frame cube, by projecting the vertices of a 3D cube
    %through the projective camera, and drawing lines betweeen the
    %resulting 2d image points
    xImCartEst = projectiveCamera(K,T,XWireFrameCart) ;
    
    % Draw a wire frame cube using data XWireFrameCart
    
    hold on ;
    set(gcf,'Color',[1 1 1]);
    imshow(im); axis off; axis image; hold on;
    plot(xImCartEst(1,:),xImCartEst(2,:),'r.','MarkerSize',10);
    
    for cPoint = 1:4
        %plot a green line between each pair of points
        pt2 = mod(cPoint,4)+1 ;
        pt3 = mod(cPoint+2,4)+1 ;
        plot([xImCartEst(1,cPoint) xImCartEst(1,pt2)],[xImCartEst(2,cPoint) xImCartEst(2,pt2)],'g-','LineWidth',2);
        hold on;
        plot([xImCartEst(1,cPoint+4) xImCartEst(1,pt2+4)],[xImCartEst(2,cPoint+4) xImCartEst(2,pt2+4)],'g-','LineWidth',2);
        hold on;
        plot([xImCartEst(1,cPoint) xImCartEst(1,cPoint+4)],[xImCartEst(2,cPoint) xImCartEst(2,cPoint+4)],'g-','LineWidth',2);
    end;
    
    hold off;
    drawnow;
    
%     if iFrame==1 || iFrame==50 || iFrame==120 
%         saveas(gcf, ['img' num2str(iFrame) '.jpg']) ;
%      end
    
    
end % End of loop over all frames.
% ================================================
end

function T = estimatePlanePose(xImCart,XCart,K)


%TO DO Convert Cartesian image points xImCart to homogeneous representation
xImHom = [xImCart; ones(1,size(xImCart,2))] ;

%TO DO Convert image co-ordinates xImHom to normalized camera coordinates
xCamHom = K^-1*xImHom ;

%TO DO Estimate homography H mapping homogeneous (x,y)
%coordinates of positions in real world to xCamHom.  Use the routine you wrote for
%Practical 1B.

H = calcBestHomography(XCart, xCamHom(1:2,:)) ;

%TO DO Estimate first two columns of rotation matrix R from the first two
%columns of H using the SVD

HTest = H(:,1:2) ;

[U,~,V] = svd(HTest) ;
inter = [ 1 0 ; 0 1; 0 0] ;
rotation = U*inter*(V.') ;

%TO DO Estimate the third column of the rotation matrix by taking the cross
%product of the first two columns

lastC = cross(rotation(:,1),rotation(:,2)) ;

%TO DO Check that the determinant of the rotation matrix is positive - if
%not then multiply last column by -1.

rotation = [rotation lastC] ;
deter = det(rotation);
if (deter == -1)
    rotation(:,3) = -1*rotation(:,3) ;
end

%TO DO Estimate the translation t by finding the appropriate scaling factor k
%and applying it to the third colulmn of H
inter = H(:,1:2)./rotation(:,1:2) ;
lambda = (1/6)*sum(inter(:)) ;

taux = H(:,3)/lambda ;

%TO DO Check whether t_z is negative - if it is then multiply t by -1 and
%the first two columns of R by -1.

if (taux(3,1) < 0 )
    rotation(:,1:2) = -1*rotation(:,1:2) ;
    taux = -1*taux ;
end

T = [rotation taux; 0 0 0 1] ;

end

function H = calcBestHomography(pts1Cart, pts2Cart)

%should apply direct linear transform (DLT) algorithm to calculate best
%homography that maps the points in pts1Cart to their corresonding matchin in
%pts2Cart

%****TO DO ****: replace this
M = size(pts1Cart,2) ;

%then construct A matrix which should be (10 x 9) in size
%solve Ah = 0 by calling h = solveAXEqualsZero(A); (you have to write this routine too - see below)
A = zeros(2*M,9);
index = 1 ;

for i = 1:M
    test = [0, 0,0, -pts1Cart(1,i), -pts1Cart(2,i), -1, pts2Cart(2,i).*pts1Cart(1,i), pts2Cart(2,i).*pts1Cart(2,i), pts2Cart(2,i)] ;
    A(index,:) = test ;
    A(index+1,:) = [pts1Cart(1,i) pts1Cart(2,i) 1 0 0 0 -pts2Cart(1,i)*pts1Cart(1,i) -pts2Cart(1,i)*pts1Cart(2,i) -pts2Cart(1,i)] ;
    index = index+2 ;
end

h = solveAXEqualsZero(A);
%reshape h into the matrix H

%Beware - when you reshape the (9x1) vector x to the (3x3) shape of a homography, you must make
%sure that it is reshaped with the values going first into the rows.  This
%is not the way that the matlab command reshape works - it goes columns
%first.  In order to resolve this, you can reshape and then take the
%transpose

H = reshape(h, [3,3]) ;
H = H.' ;
end

%==========================================================================
function x = solveAXEqualsZero(A)
[~,~,V] = svd(A) ;
N = size(V,2) ;
x = V(:,N) ;
end

function xImCart = projectiveCamera(K,T,XCart)

%TO DO convert Cartesian 3d points XCart to homogeneous coordinates XHom
XHom = [XCart; ones(1,size(XCart,2))] ;

%TO DO apply extrinsic matrix to XHom to move to frame of reference of
%camera
XHom = T*XHom ;

%TO DO project points into normalized camera coordinates xCamHom by (achieved by
%removing fourth row)
xCamHom =  XHom(1:3,:) ;

%TO DO move points to image coordinates xImHom by applying intrinsic matrix
xImHom = K*xCamHom ;

%TO DO convert points back to Cartesian coordinates xImCart
xImCart = xImHom(:,:)./repmat(xImHom(3,:),3,1) ;

xImCart = xImCart(1:2, :);
end

