function clusters = read_cluster(filename)
    % Open file
    fid = fopen(filename, 'r');
    if fid == -1
        error('Could not open file: %s', filename);
    end
    % Read entire file into cell array
    raw_lines = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
    fclose(fid);
    raw_lines = raw_lines{1}; % Extract cell array of lines
    % Initialize variables
    cluster_id = 0;
    clusters = struct([]);
    % Process lines
    for i = 1:length(raw_lines)
        line = strtrim(raw_lines{i});
        % Identify new cluster
        if startsWith(line, '#')
            cluster_id = cluster_id + 1;
            clusters(cluster_id).id = cluster_id;
            clusters(cluster_id).events = [];
            cluster_id
        elseif ~isempty(line) && cluster_id > 0
            codes = split(line, ',');
            % Parse data
            ot = codes{1};                      % Origin time as string
            lat = str2double(codes{2});         % Latitude as float
            lon = str2double(codes{3});         % Longitude as float
            dep = str2double(codes{4});         % Depth as float
            mag = str2double(codes{5});         % Magnitude as float
            evid = str2double(codes{6});        % Event ID as integer
            % Append to cluster
            clusters(cluster_id).events = [clusters(cluster_id).events; {ot, lat, lon, dep, mag, evid}];
        end
    end
    
    % Convert events into structured fields for easy access
    for i = 1:length(clusters)
        if ~isempty(clusters(i).events)
            clusters(i).ot = clusters(i).events(:, 1);
            clusters(i).lat = clusters(i).events(:, 2);
            clusters(i).lon = clusters(i).events(:, 3);
            clusters(i).dep = clusters(i).events(:, 4);
            clusters(i).mag = clusters(i).events(:, 5);
            clusters(i).evid = clusters(i).events(:, 6);
        else
            clusters(i).ot = [];
            clusters(i).lat = [];
            clusters(i).lon = [];
            clusters(i).dep = [];
            clusters(i).mag = [];
            clusters(i).evid = [];
        end
    end
end
