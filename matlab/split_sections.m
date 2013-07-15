function split_sections(expname, sections, result, basepath)
% split_sections(experiment name, section, result, [basepath])
%%% Splits an original BeadTracker2 trace file into pieces according to its section file.
%%% Combines all the frames from a set of sections into one file (result).
%%% sections: ordered array of sections such as 10:20. Zero-based, like the sections file itself!
%%% result: result filename.
%%% tracespath: path in which the trace, motors and section files will be found

    if nargin<4
        basepath='.';
    end
    
    sectfilename = sprintf('%s\\%s_sections.txt', basepath, expname);
    sectlist = dlmread(sectfilename);
    
    if nargin<2
        sections=sectlist(:,1);
    end
    if nargin<3
        result=sprintf('%s\\%s_sect%s.txt', basepath, expname, sprintf('_%d',sections));
    end
    
    tracefile = sprintf('%s\\%s.txt', basepath, expname);
    fprintf('Opening trace file %s\n', tracefile);
    fid = fopen(tracefile, 'r');
    fprintf('Result file: %s\n', result);
    fout = fopen(result,'w');
    
    cursect = 1;
    
    linenum = 0;
    while 1
        line = fgetl(fid);
        if ~ischar(line), break, end

        s_index = sections(cursect)+1;
        % see if we should save this frame's data
        frame = sectlist(s_index, 2);
        if linenum >= frame
            linedata=textscan(line, '%f'); 
            linedata=linedata{:}; % convert cell matrix to regular matrix
            
            fprintf(fout, '%.6f\t', linedata);
            fprintf(fout, '\n');
        end
        
        if s_index < length(sectlist) && sectlist(s_index+1,2) <= linenum
            cursect = cursect+1;
            if cursect == length(sections)+1, break, end;
            fprintf('section: %d\n', sections(cursect));
        end
        linenum =linenum+1;
        if (mod(linenum,1000)==0)
            fprintf('Processing frame %d\n', linenum);
        end
    end
   
    fclose(fout);
    fclose(fid);

end