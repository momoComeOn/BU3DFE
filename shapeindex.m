function [s, sflag] = shapeindex(flag,kmin,kmax)
% s = shapeindex(flag, kmax, kmin);
%
% Written by Dirk Colbry
% 11-03-03
%
% This function calculates the shape ratio of all the points from kmax, kmin.


[nr nc]=size(flag);
s = zeros(nr, nc);
for i=1:nr
%    fprintf(1,'Shape row %03d of %03d\r',i,nr);
    for j=1:nc
        if flag(i,j)            
			if(kmin(i,j) ~= kmax(i,j))
                S = 1/2 - 1/pi*atan((kmin(i,j) + kmax(i,j))/abs(kmin(i,j)-kmax(i,j)));
			else
                S = Inf;
			end                                   
            s(i,j) = S;                                    
        else
            s(i,j) = Inf;
        end
        
    end
end

sflag = ~isinf(s);

%fprintf(1,'...Done.\n');