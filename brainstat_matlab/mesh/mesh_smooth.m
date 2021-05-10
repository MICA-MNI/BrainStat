function Y = mesh_smooth( Y, surf, FWHM );
% MESH_SMOOTH    Smooths surface data by repeatedly averaging over edges. Y =
%   Y = MESH_SMOOTH=(Y, surf, FWHM) smooths data Y on surface surf using the
%   approximate FWHM of Gaussian smoothing filter. Y is a
%   sample-by-vertex-by-variate matrix; surf is a surface in SurfStat format
%   i.e. a struct containing a 'tri' field for triangle indices or a 'lat' field
%   for lattices; FWHM is a scalar expressed in mesh units. 

niter=ceil(FWHM^2/(2*log(2)));

if isnumeric(Y)
    [n,v,k]=size(Y);
    isnum=true;
else
    Ym=Y;
    s=Ym.Format{2};
    if length(s)==2
        s=s([2 1]);
        k=1;
    else
        s=s([3 1 2]);
        k=s(3);
    end
    n=s(1);
    v=s(2);
    isnum=false;
end

edg=mesh_edges(surf);

Y1=accumarray(edg(:,1),2,[v 1])'+accumarray(edg(:,2),2,[v 1])';

if n>1
    fprintf(1,'%s',[num2str(n) ' x ' num2str(k) ' surfaces to smooth, % remaining: 100 ']);
end
n10=floor(n/10);
for i=1:n
    if rem(i,n10)==0
        fprintf(1,'%s',[num2str(100-i/n10*10) ' ']);
    end
    for j=1:k
        if isnum
            Ys=squeeze(Y(i,:,j));
            for iter=1:niter
                Yedg=Ys(edg(:,1))+Ys(edg(:,2));
                Ys=(accumarray(edg(:,1),Yedg',[v 1]) + ...
                    accumarray(edg(:,2),Yedg',[v 1]))'./Y1;
            end
            Y(i,:,j)=Ys;
        else
            if length(s)==2
                Y=Ym.Data(1).Data(:,i);
            else
                Y=Ym.Data(1).Data(:,j,i);
            end            
            for iter=1:niter
                Yedg=Y(edg(:,1))+Y(edg(:,2));
                Y=(accumarray(edg(:,1),Yedg',[v 1]) + ...
                    accumarray(edg(:,2),Yedg',[v 1]))'./Y1;
            end
            if length(s)==2
                Ym.Data(1).Data(:,i)=Y;
            else
                Ym.Data(1).Data(:,j,i)=Y;
            end            
        end
    end
end
if n>1
    fprintf(1,'%s\n','Done');
end
if ~isnum
    Y=Ym;
end

return
end