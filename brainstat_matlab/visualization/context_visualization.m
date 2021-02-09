function h = context_visualization(decoder, field)

% Get surfaces. 
switch decoder.surface
    case 'conte69'
        [surf_lh, surf_rh] = load_conte69(); % Currently sourced from BrainSpace
    case 'fsaverage5'
        % TO-DO
    otherwise
        error('Unknown surface type.');
end

% Get some figure properties.
field = lower(field); 
names = replace(decoder.(field).names,'_',' ');
xMax = abs(max(decoder.(field).r));

% Build the horizontal bar plot. 
h.figure = figure('Units','Normalized','Position',[0, 0, .5, 1],'Color','w');
h.axes(1) = axes('position',[0.3 0.15 0.4 0.8]);
h.barh = barh(decoder.(field).r);
set(h.axes(1)                                       , ...
    'YTick'             , 1:numel(decoder.(field).names), ...
    'YTickLabels'       , names                     , ...
    'XTick'             , [-xMax,0,xMax]            , ...
    'XLim'              , [-xMax,xMax]              , ...
    'FontName'          , 'DroidSans'               , ...
    'FontSize'          , 16                        , ...
    'Box'               , 'off'                     );
set(h.barh                                          , ...
    'FaceColor'         , [.7 .7 .7]                , ...
    'BarWidth'          , .6                        );
end

