function nn = resnet52_2stream()
% Concat two model.
if(~exist('net.mat'))
    net1 = resnet52_market_2stream_fc_identify_solely();
    net1.removeLayer('top5err');
    net2 = resnet52_market_2stream_fc_identify_solely(); %imagenet
    net2.removeLayer('top5err');
    %change name
    for i = 1:numel(net2.layers)
        net2.renameLayer(net2.layers(i).name,sprintf('%s_2',net2.layers(i).name));
    end
    for i = 1:numel(net2.vars)
        net2.renameVar(net2.vars(i).name,sprintf('%s_2',net2.vars(i).name));
    end
    nn = concat_2net(net1,net2);
    %net_struct = nn.saveobj();
    %save('net.mat','net_struct');
else
    load('net.mat');
    nn = dagnn.DagNN.loadobj(net_struct);
end
% *****************************************************************************


nn.initParams();


