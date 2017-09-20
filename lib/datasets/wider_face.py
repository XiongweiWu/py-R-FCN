# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from fast_rcnn.config import cfg

class wider_face(imdb):
    def __init__(self, image_set, year, face_path=None):
        imdb.__init__(self, 'widface_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._face_path = self._get_default_path() if face_path is None \
                            else face_path
        self._data_path = os.path.join(self._face_path, 'FACE' + self._year)
        self._classes = ('__background__', # always index 0
                         'face')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'
	
        # WIDERFACE specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_blur'    : True,
                       'use_expr'    : True,
                       'use_illu'    : True,
                       'use_invalid' : False,
                       'use_occ'     : True,
                       'use_pose'    : True,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._face_path), \
                'Wider face path does not exist: {}'.format(self._face_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)
    
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
   	    #WIDERFACE/FACE2017/WIDER_<image_set>/images/--0---name/XXXX.jpg
        image_path = os.path.join(self._data_path, 
				  'WIDER_{}'.format(self._image_set), 'images',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        # Example path to image set file:
        # /WIDERFACE/FACE2007/wider_face_split/wider_face_val.mat
        # image path : WIDERFACE/FACE2017/images/0--paradim/01---ssda.jpg
        
        # load from gt list is too slow, instead we load from mat file
        image_index = []
        image_set_file = os.path.join(self._data_path, 'wider_face_split', \
                'wider_face_{}.mat'.format(self._image_set))	
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        raw_data = sio.loadmat(image_set_file) # dic
        evt_names = raw_data['event_list']
        img_lists = raw_data['file_list']
        total_cont = 0
        for i in range(img_lists.shape[0]):
            evt_name = evt_names[i][0].ravel()[0]
            img_names = img_lists[i][0].ravel()
            print "Event {} has {} images".format(evt_name, img_names.shape[0])
            total_cont += img_names.shape[0]
            for j in range(img_names.shape[0]):
                img_name = img_names[j][0]
                image_index.append(os.path.join(evt_name, img_name))

        print "{} has totally {} images".format(self._image_set, total_cont)
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'WIDERFACE')

    def _read_annfile(self):
        #filename = "wider_face_{}_bbx_gt.txt".format(self._image_set)
        filename = os.path.join(self._data_path,'wider_face_split', \
                "wider_face_{}_bbx_gt.txt".format(self._image_set))
        assert os.path.exists(filename), \
                'Path does not exist: {}'.format(filename)
        
        annotations = []
        diffs = 0
        with open(filename, 'r') as f:
            recs = [x.strip() for x in f.readlines()]
            for img in self._image_index:
                print len(annotations)
                img = img + self._image_ext
                try:
                    idx = recs.index(img)
                except:
                    print "no annotation info for {}".format(img)
                    exit()

                diff_count = 0
                num_objs = int(recs[idx+1])
                boxes = np.zeros((0, 4), dtype=np.uint16)
                gt_classes = np.zeros((0), dtype=np.int32)
                overlaps = np.zeros((0, 2), dtype=np.float32)
                seg_areas = np.zeros((0), dtype=np.float32)
                # consider diff filter here        
                for ix in range(num_objs):
                    obj_idx = idx+1 + ix+1 # +0+1
                    # x1 y1 w h blur exp ill invlaid occ pose
                    obj = recs[obj_idx].split(' ')
                
                    blur = int(obj[4])
                    expr = int(obj[5])
                    illu = int(obj[6])
                    invalid = int(obj[7])
                    occ = int(obj[8])
                    pose = int(obj[9])
                
                    if (not self.config['use_blur'] and blur != 0) or (not self.config['use_expr'] and expr != 0) \
                        or (not self.config['use_illu'] and illu !=0) or (not self.config['use_invalid'] and invalid != 0) \
                        or (not self.config['use_occ'] and occ != 0) or (not self.config['use_pose'] and pose != 0):
                            diff_count += 1
                            continue
                    # Make pixel indexes 0-based
                    x1 = float(obj[0]) 
                    y1 = float(obj[1]) 
                    x2 = x1 + float(obj[2]) 
                    y2 = y1 + float(obj[3]) 
                    assert x1<=x2, "x1:{} is larger than x2:{}".format(x1, x2)
                    assert y1<=y2, "y1:{} is larger than y2:{}".format(y1, y2)
                    #
                    cls = self._class_to_ind['face']
                    boxes = np.vstack([boxes,  [x1, y1, x2, y2]])
                    gt_classes = np.hstack([gt_classes, cls])
                    overlaps = np.vstack([overlaps, [0, 1]])
                    seg_areas = np.hstack([seg_areas , (x2 - x1 + 1) * (y2 - y1 + 1)])

                overlaps = scipy.sparse.csr_matrix(overlaps)
                assert diff_count + boxes.shape[0] == num_objs
                annotations.append({'boxes' : boxes,
                    'gt_classes': gt_classes,
                    'gt_overlaps' : overlaps,
                    'flipped' : False,
                    'seg_areas' : seg_areas})
                diffs += diff_count
        print "{} difficult faces are removed".format(diffs)
        #print len(self._image_index)
        assert len(annotations) == len(self._image_index)
        return annotations

	
    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        anno_file  = os.path.join(self.cache_path, self.name + '_annonation.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
		

        if os.path.exists(anno_file):
            with open(anno_file, 'rb') as fid:
                self._annotation = cPickle.load(fid)
            print '{} annotation loaded from {}'.format(self.name, anno_file)
        else:
            self._annotation = self._read_annfile()
            with open(anno_file, 'wb') as fid:
                cPickle.dump(self._annotation, fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote annotation to {}'.format(anno_file)
	

        gt_roidb = [self._annotation[index]
                    for index in xrange(len(self._image_index))]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb


    def rpn_roidb(self):
        if int(self._year) == 2017 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb


    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_widface_annotation(self, index):
        idx = self._image_index.index(index)
        roidb =  self._annotation[idx]
        return roidb


    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_widface_results_file_template(self):
        # WIDERFACE/results/FACE2017/<comp_id>_det_face.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._wider_path,
            'results',
            'FACE' + self._year,
            filename)
        return path

    def _write_widface_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir = 'output'):
        print "no eval yet"
        exit()
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print "no eval yet"
        exit()
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_widface_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.wider_face import wider_face
    d = wider_face('val', '2017')
    res = d.roidb
    from IPython import embed; embed()
