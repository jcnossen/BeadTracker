function [beadx, beady, beadz, timestamps, frameinfo, axisnames] = qtrk_read_shittybin(filename, frames, beads, refbead)
% Read XYZ traces from a binary trace output produced by TweezerTracker
% [beadx, beady, beadz, timestamps, frameinfo, axisnames] =  ...
%    qtrk_read_bin_trace(filename, frames, beads, refbead)
%
% Example: qtrk_read_bin_trace('test.bin', 20:100);  % reads frame 20 to 100 for all beads from test.bin
%
% beadx,beady,beadz: XYZ data, beads are mapped to columns, frames are mapped to rows
% frames: list of frame numbers, or -1 for all frames
% beads: List of beads (1-based indexing) (-1 for all)
% refbead: Index of reference bead (-1 to disable reference bead subtraction)
    if nargin<4
        refbead=-1; % no reference bead subtraction
    end
    
    if nargin<5
        oldver=0;
    end
    
    [f_nframes, nbeads, f_ninfocol, axisnames, data_offset] = sizeof_bin_trace(filename);
    if f_nframes == 0, return, end %something went wrong
    
    fprintf('%s has %d beads and %d frames.\n', filename, nbeads, f_nframes);
    
    if nargin<3 || ( isscalar(beads) &&  beads<0)
        beads = 1:nbeads; % load all beads
    end
    
    if nargin<2 || (isscalar(frames) && frames < 0)
        frames=1:f_nframes;
    end
   
    beadx = zeros(length(frames),length(beads));
    beady = beadx;
    beadz = beadx;    
    timestamps = zeros(length(frames),1);
    frameinfo = zeros(length(frames),f_ninfocol);
    
    fid = fopen(filename);

    haveErrors=1;
    bytesPerFrame = 4 + 8 + f_ninfocol * 4 + nbeads * 4 * 3;
    if haveErrors, bytesPerFrame = bytesPerFrame + nbeads * 4; end;
    
    lastFrame=-1;
    for k=1:length(frames)
        
        f=frames(k);
        if f~=lastFrame+1
            filepos = data_offset + bytesPerFrame * (f-1);
            fseek(fid, filepos, -1);
        end
        
        frame_id = fread(fid, 1, 'uint32');
        assert(frame_id == f-1);
        timestamps(k) = fread(fid, 1, 'double');
        frameinfo(k, :) = fread(fid, [1 f_ninfocol], 'single');
        
        xyz = fread(fid, nbeads * 3, 'single');
        errorvals = fread(fid, nbeads, 'uint32');
        
		% Read all beads
        bx = xyz( (0:nbeads-1) * 3 + 1 );
        by = xyz( (0:nbeads-1) * 3 + 2 );
        bz = xyz( (0:nbeads-1) * 3 + 3 );

		% Take the subset of beads that we want
        beadx(k, :) = bx (beads);
        beady(k, :) = by (beads);
        beadz(k, :) = bz (beads);
        
		% Subtract refbead if set
        if refbead >= 0
            beadx(k, :) = beadx(k,:) - bx(refbead);
            beady(k, :) = beady(k,:) - by(refbead);
            beadz(k, :) = beadz(k,:) - bz(refbead);
        end
        
        lastFrame=f;
    end
    
    fclose(fid);
end

function [nframes, nbeads, ninfocol, colnames, data_offset] = sizeof_bin_trace(filename)
    fid = fopen(filename,'r', 'ieee-le'); % little endian byte order
    if fid<0,
        fprintf('can ''t open file %s', filename);
        nframes=0;
        return
    end
    
    if nargin<2, oldver=0; end;

    version = int32(fread(fid, 1, 'int32'));
    nbeads = int32(fread(fid, 1, 'int32'));
    ninfocol = int32(fread(fid, 1, 'int32'));
    data_offset = int32(fread(fid, 1, 'int32'));
    
    if (data_offset == 1234)  % broken bin
        data_offset = ninfocol;
        ninfocol = 7;
    end
    
    for k=1:ninfocol
        colnames{k} = read_zero_term_string(fid);
    end

    % figure out the size of this file and compute number of frames from it
    fseek(fid, 0, 'eof');
    file_size = ftell(fid);
    bytesPerFrame = 4 + 8 + ninfocol * 4 + nbeads * 4 * 3 + nbeads*4;
    nframes = int32((file_size - data_offset) / bytesPerFrame);
    fclose(fid);
end

function s = read_zero_term_string(fid)
    s= char;
    while ~feof(fid)
        ch = fread(fid, 1, 'int8');
        if ch == 0, break, end
        s = [s ch];
    end
end
