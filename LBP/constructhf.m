function featurevectors=constructhf(inputvectors, map)

    n=map.samples;
    FVLEN=(n-1)*(floor(n/2)+1)+3;
    featurevectors=zeros(size(inputvectors,1),FVLEN);
    
    k=1;
    for j=1:length(map.orbits)
        b=inputvectors(:,map.orbits{j}+1);
        if(size(b,2) > 1)
            b=fft(b')';
            b=abs(b);
            b=b(:,1:(floor(size(b,2)/2)+1));
        end
        featurevectors(:,k:k+size(b,2)-1)=b;
        k=k+size(b,2);
    end
