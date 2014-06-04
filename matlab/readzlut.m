function mat = readzlut(filename, normalize)

    if (nargin <1)
        filename = '10x.radialzlut';
    end

    dim=3;
    elemtype = 'single';

    fid = fopen(filename,'r', 'ieee-le'); % little endian

    if fid ~= -1
        s = fread(fid, dim, 'int32');
        
        nbeads = s(1);
        nplanes = s(2);
        radialsteps = s(3);
        
        mat = zeros([nplanes radialsteps nbeads]);
        
        %mat = fread(fid, s, elemtype);
        %d = zeros(s(2),s(1));
        %for k=1:s(2)
        
        for k=1:nbeads
            mat(:,:,k) = fread(fid , [radialsteps nplanes], elemtype)';
        end
        
        fclose(fid);
        
        if nargin>1 && normalize
            for k=1:nbeads
                for p=1:nplanes
                    row = mat(p, :, k);
                    mat(p, :, k) = (row - min(row(:))) ./ (max(row(:)) - min(row(:)));
                end
            end
        end
    end


end