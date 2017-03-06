function r=HW2_Practical9d( templateMetaFileName )

LoadVideoFrames

load( templateMetaFileName );


numParticles = 50;
weight_of_samples = ones(numParticles,1);

% TO DO: normalize the weights (may be trivial this time)
weight_of_samples = weight_of_samples./sum(weight_of_samples(:));

% Initialize which samples from "last time" we want to propagate: all of
% them!:
samples_to_propagate = [1:numParticles]';

numDims_w = 2;

particles_old = repmat([minY minX], numParticles, 1 ) + 5*rand( numParticles, numDims_w );

r = zeros(numFrames, numDims_w);

for( iTime = 1:numFrames )
    cum_hist_of_weights = cumsum(weight_of_samples) ;

    samples_to_propagate = zeros(numParticles,1);

    some_threshes = rand(numParticles,1);

    for sampNum = 1:numParticles
        thresh = some_threshes(sampNum);
        for index = 1:numParticles
            if( cum_hist_of_weights(index) > thresh )
                break;
            end;
        end;
        samples_to_propagate(sampNum) = index;
    end;

    particles_new = zeros( size(particles_old) );
    for particleNum = 1:numParticles

        noise = 10*randn(1,size(particles_old,2)) ;
        particles_new(particleNum,:) = particles_old( samples_to_propagate(particleNum),: ) + noise ;
        particles_new(particleNum,:) = round(  particles_new(particleNum,:)  ); % Round the particles_new to simplify Likelihood evaluation.
    end;

    Im2 = double( Imgs{iTime} );

    for particleNum = 1:numParticles

        particle = particles_new(particleNum,:);

        s = size(pixelsTemplate);
        inFrame = particle(1) >= 1.0   &&  particle(1)+ s(1) <= imgHeight && ...
                particle(2) >= 1.0   &&  particle(2) + s(2) <= imgWidth;
        if( inFrame )
            minX = particles_new(particleNum,2);
            minY = particles_new(particleNum,1);

            weight_of_samples(particleNum) = ...
                MeasurePatchSimilarityHere( Im2, pixelsTemplate, minY, minX );
        else
            weight_of_samples(particleNum) = 0.0;
        end;

    end;

    weight_of_samples = weight_of_samples./sum(weight_of_samples(:));

    weightRep = repmat(weight_of_samples, 1,2) ;
    
    weightedAve = particles_new.*weightRep ;
    
    weightedAve = sum(weightedAve,1) ;
    
    middleOfTrackedPatch = weightedAve + patchOffset;
    r(iTime,:) = middleOfTrackedPatch; 

    particles_old = particles_new;
    clear particles_new;

end;
