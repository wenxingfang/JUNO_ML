import ROOT as rt
import numpy as np
import h5py
import math
import sys
import argparse
import gc
rt.gROOT.SetBatch(rt.kTRUE)
rt.TGaxis.SetMaxDigits(3);
# npe will be save one dataframe for each r,theta
def get_parser():
    parser = argparse.ArgumentParser(
        description='Produce training samples for JUNO study. ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--batch_size', action='store', type=int, default=5000,
                        help='Number of event for each batch.')
    parser.add_argument('--input', action='store', type=str, default='',
                        help='input root file.')
    parser.add_argument('--output', action='store', type=str, default='',
                        help='output hdf5 file.')
    parser.add_argument('--r_scale', action='store', type=float, default=17700,
                        help='r normalization')
    parser.add_argument('--theta_scale', action='store', type=float, default=180,
                        help='theta scale.')
    parser.add_argument('--phi_scale', action='store', type=float, default=180,
                        help='dphi scale.')

    return parser

def root2hdf5 (batch_size, tree, start_event, out_name, id_dict, x_max, y_max):

    hf = h5py.File(out_name, 'w')
    df = np.full((batch_size, y_max+1, x_max+1, 2), 0, np.float32)
    df_true = np.full((batch_size, 3+3+1), 0, np.float32)##position, momenta, KE 
    for ie in range(start_event, batch_size+start_event):
        tree.GetEntry(ie)
        tmp_dict = {}
        tmp_firstHitTime_dict = {}
        init_x    = getattr(tree, "init_x")
        init_y    = getattr(tree, "init_y")
        init_z    = getattr(tree, "init_z")
        init_px    = getattr(tree, "init_px")
        init_py    = getattr(tree, "init_py")
        init_pz    = getattr(tree, "init_pz")
        pmtID     = getattr(tree, "pmt_id")
        hittime   = getattr(tree, "pmt_hit_time")
        init_KE = math.sqrt( init_px*init_px + init_py*init_py + init_pz*init_pz + 0.511*0.511) - 0.511
        df_true[ie-start_event, 0] = init_x
        df_true[ie-start_event, 1] = init_y
        df_true[ie-start_event, 2] = init_z
        df_true[ie-start_event, 3] = init_px
        df_true[ie-start_event, 4] = init_py
        df_true[ie-start_event, 5] = init_pz
        df_true[ie-start_event, 6] = init_KE
        for i in range(0, len(pmtID)):
            ID     = pmtID[i]
            if ID not in id_dict:continue
            if ID not in tmp_dict:
                tmp_dict[ID] = 1
            else:
                tmp_dict[ID] = tmp_dict[ID] + 1
            if ID not in tmp_firstHitTime_dict:
                tmp_firstHitTime_dict[ID] = hittime[i]
            else:
                tmp_firstHitTime_dict[ID] = hittime[i] if hittime[i] < tmp_firstHitTime_dict[ID] else tmp_firstHitTime_dict[ID]
        for iD in id_dict:
            ix     = id_dict[iD][0]
            iy     = id_dict[iD][1]
            tmp_npe = 0
            tmp_firstHitTime = 0
            if iD in tmp_dict:
                tmp_npe          = tmp_dict[iD]
                tmp_firstHitTime = tmp_firstHitTime_dict[iD]
            df[ie-start_event,iy,ix,0] = tmp_npe
            df[ie-start_event,iy,ix,1] = tmp_firstHitTime

    hf.create_dataset('data', data=df)
    hf.create_dataset('label', data=df_true)
    hf.close()
    print('saved %s'%out_name, 'with data shape=',df.shape,', label=', df_true.shape)
    if Draw_data:
        for i in range(10):
            tmp_df_0 = df[i,:,:,0]
            tmp_df_1 = df[i,:,:,1]
            tmp_init_x = df_true[i,0] 
            tmp_init_y = df_true[i,1] 
            tmp_init_z = df_true[i,2] 
            tmp_init_px = df_true[i,3] 
            tmp_init_py = df_true[i,4] 
            tmp_init_pz = df_true[i,5] 
            tmp_init_KE = df_true[i,6] 
            draw_data(outname='npe_pos_%f_%f_%f_mom_%f_%f_%f_KE_%f'%(tmp_init_x, tmp_init_y, tmp_init_z, tmp_init_px, tmp_init_py, tmp_init_pz, tmp_init_KE), x_max=x_max, y_max=y_max, df=tmp_df_0)
            draw_data(outname='Ftime_pos_%f_%f_%f_mom_%f_%f_%f_KE_%f'%(tmp_init_x, tmp_init_y, tmp_init_z, tmp_init_px, tmp_init_py, tmp_init_pz, tmp_init_KE), x_max=x_max, y_max=y_max, df=tmp_df_1)



def get_pmt_theta_phi(file_pos, sep, i_id, i_theta, i_phi):
    id_dict = {}
    theta_list = []
    phi_list = []
    f = open(file_pos,'r')
    lines = f.readlines()
    for line in lines:
        items = line.split()
        ID    = float(items[i_id])
        ID    = int(ID)
        theta = float(items[i_theta])
        phi   = float(items[i_phi])
        #phi   = int(phi) ## otherwise it will be too much
        if theta not in theta_list:
            theta_list.append(theta)
        if phi not in phi_list:
            phi_list.append(phi)
        if ID not in id_dict:
            id_dict[ID]=[theta, phi]
    return (id_dict, theta_list, phi_list)

def get_pmt_x_y_z_theta_phi(file_pos, i_id, i_x, i_y, i_z, i_theta, i_phi):
    id_dict = {}
    f = open(file_pos,'r')
    lines = f.readlines()
    for line in lines:
        items = line.split()
        ID    = float(items[i_id])
        ID    = int(ID)
        x     = float(items[i_x])
        y     = float(items[i_y])
        z     = float(items[i_z])
        theta = float(items[i_theta])
        phi   = float(items[i_phi])
        if ID not in id_dict:
            id_dict[ID]=[x, y, z, theta, phi]
    return id_dict


def do_plot2d(hist,out_name,title):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    #h_corr.Draw("COLZ")
    #h_corr.LabelsDeflate("X")
    #h_corr.LabelsDeflate("Y")
    #h_corr.LabelsOption("v")
    hist.SetStats(rt.kFALSE)
    hist.GetXaxis().SetTitle(title['X'])
    hist.GetYaxis().SetTitle(title['Y'])
    hist.GetXaxis().SetTitleOffset(1.2)
    hist.Draw("COLZ")
    canvas.SaveAs("%s.png"%(out_name))
    del canvas
    gc.collect()


def vertexRec_map(file_pos, i_id, i_x, i_y, i_z, i_theta, i_phi):
    max_width = 230
    x_shift = 115
    f = open(file_pos,'r')
    lines = f.readlines()
    pmt_zi = 0
    z_index = 0
    First = True
    lx_ly_id_dict = {}
    id_lx_ly_dict = {}
    for line in lines:
        items = line.split()
        ID    = float(items[i_id])
        ID    = int(ID)
        x     = float(items[i_x])
        y     = float(items[i_y])
        z     = float(items[i_z])
        phi   = float(items[i_phi])
        if phi > 180: phi = phi-360
        local_r = math.sqrt(x*x + y*y)
        r       = math.sqrt(x*x + y*y + z*z)
        if First :
            pmt_zi = z
            First = False

        # set the index of z axis
        if int(z) == int(pmt_zi):
            pass
        else:
            z_index += 1
            pmt_zi = z

        #shift the first and last 21 pmts to avoid overlap
        if ID == 7 or ID == 17606: 
            z_index += 1

        #lx = int(np.floor((phi * (local_r / r) / (np.pi * 2.0)) * 230)) + 150
        #print('phi=',phi,',local_r=',local_r,',r=',r)
        #lx = int( max_width* phi * local_r / (r * 360) )
        lx = round( max_width* phi * local_r / (r * 360) ) + x_shift
        ly = z_index
        if lx not in lx_ly_id_dict:
            lx_ly_id_dict[lx] = {}
        if ly not in lx_ly_id_dict[lx]:
            lx_ly_id_dict[lx][ly] = ID
        else:
            print('overlap:ID=',ID,',ori ID=',lx_ly_id_dict[lx][ly],',lx=',lx,',ly=',ly,',x=',x,'y,=',y,'z=',z)
        id_lx_ly_dict[ID]=[lx,ly] 
    return id_lx_ly_dict

def draw_map(id_dict):
    x_min = 999
    x_max = -1
    y_min = 999
    y_max = -1
    for i in id_dict:
        x = id_dict[i][0]
        y = id_dict[i][1]
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y
    print('x_min=',x_min,',x_max=',x_max,',y_min=',y_min,',y_max=',y_max)
    h_map = rt.TH2F('map','',x_max+2, -1, x_max+1, y_max+2, -1, y_max+1)
    for i in id_dict:
        x = id_dict[i][0]
        y = id_dict[i][1]
        h_map.SetBinContent(x+1, y+1, i) 
    do_plot2d(hist=h_map, out_name='%s/id_map'%plots_path, title={'X':'x(phi)','Y':'y(z)'})
    return (x_min, x_max, y_min, y_max)

def draw_data(outname, x_max, y_max, df):
    h_map = rt.TH2F('map_%s'%outname,'',x_max+2, -1, x_max+1, y_max+2, -1, y_max+1)
    for iy in range(df.shape[0]):
        for ix in range(df.shape[1]):
            h_map.SetBinContent(ix+1, iy+1, df[iy][ix]) 
    do_plot2d(hist=h_map, out_name='%s/map_%s'%(plots_path, outname), title={'X':'x(phi)','Y':'y(z)'})


if __name__ == '__main__':

    Draw_data = False
    plots_path = '/junofs/users/wxfang/FastSim/GAN/JUNO/Reco_Energy/plots/'
    large_PMT_pos = '/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J20v2r0-Pre0/offline/Simulation/DetSimV2/DetSimOptions/data/PMTPos_Acrylic_with_chimney.csv'#FIXME to J20v2r0-Pre0 
    L_id_dict = vertexRec_map(file_pos=large_PMT_pos, i_id=0, i_x=1, i_y=2, i_z=3, i_theta=4, i_phi=5)
    print('L_id_dict=',len(L_id_dict))
    (x_min, x_max, y_min, y_max) = draw_map(L_id_dict)
    assert ( x_min==0 and y_min==0 )
    ###########################################################
    parser = get_parser()
    parse_args = parser.parse_args()

    batch_size = parse_args.batch_size
    filePath = parse_args.input
    outFileName= parse_args.output
    r_scale = parse_args.r_scale
    theta_scale = parse_args.theta_scale
    phi_scale = parse_args.phi_scale

    treeName='evt'
    chain =rt.TChain(treeName)
    chain.Add(filePath)
    tree = chain
    totalEntries=tree.GetEntries()
    if batch_size < 0 : 
        batch_size = totalEntries
    batch = int(float(totalEntries)/batch_size)
    print ('total events=%d, batch_size=%d, batchs=%d, last=%d'%(totalEntries, batch_size, batch, totalEntries%batch_size))
    start = 0
    for i in range(batch):
        out_name = outFileName.replace('.h5','_batch%d_N%d.h5'%(i, batch_size))
        root2hdf5 (batch_size, tree, start, out_name, L_id_dict, x_max, y_max)
        start = start + batch_size
    print('done')  
