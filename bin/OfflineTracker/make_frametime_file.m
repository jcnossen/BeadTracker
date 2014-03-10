function files=make_frametime_file(dirname, timestep)
    if nargin<1
        dirname='D:/jcnossen1/2012-11-23/tmp_010';
    end        
    if nargin<2
        timestep=50;
    end

    files = dir(dirname);
    files = files(~[files.isdir]);
    which = arrayfun(@(e) length(e{:}), strfind({files.name},'.jpg'));
    files = files(which~=0);
    nums = zeros(length(files),2);
    for k=1:length(files)
        nums(k,1) = str2double(files(k).name(1:8));
        nums(k,2) = k*timestep;
    end
    fn = [dirname '/frametime.txt'];
    dlmwrite(fn, nums, 'delimiter','\t','newline','pc');
    fprintf('%s written.\n',fn);
end