import math
from matplotlib import collections as mc
import matplotlib.pyplot as plt
import numpy as np

import cv2


class MatplotlibRenderer():
    """Renderer module to visualize 2d elements and contact points, using matplotlib
    """

    def __init__(self, elems, contps):
        """Constructor method

        :param elems: Elements of the model
        :type elems: dict
        :param contps: Contact points of the model
        :type contps: dict
        """
        self.elems = elems
        self.contps = contps

        self.xlim = [0, 100]
        self.ylim = [0, 100]

    def set_plot_limits(self, xlim, ylim):
        self.xlim = xlim
        self.ylim = ylim

    def plot_model(self, filename='displace_elements', invert_y=False, color_type='custom'):
        """Plot elements and contact points

        :param factor: Amplification of the plotted displacement, defaults to 1
        :type factor: int, optional
        """
        tab20 = plt.get_cmap('tab20', 20)
        fig, ax = plt.subplots()

        # element center
        for key, value in self.elems.items():
            if value.type.startswith('stone'):
                _center_color = 'r'
                _marker = 'p'
            elif value.type == 'mortar':
                _center_color = 'g'
                _marker = 'o'
            elif value.type == 'ground':
                _center_color = 'k'
                _marker = '^'
            else:
                _center_color = 'b'
                _marker = 'v'
            ax.scatter(value.center[0],
                       value.center[1], s=0.1, c=_center_color, marker=_marker)

        contfs = dict()
        for p in self.contps.values():
            if color_type == 'custom':
                if self.elems[p.cand].type == 'mortar' and self.elems[p.anta].type == 'mortar':
                    color_id = 1
                elif (self.elems[p.cand].type.startswith('stone') and self.elems[p.anta].type == 'mortar') or\
                        (self.elems[p.cand].type == 'mortar' and self.elems[p.anta].type.startswith('stone')):
                    color_id = 2
                elif self.elems[p.cand].type == 'ground' or self.elems[p.anta].type == 'ground' or\
                        self.elems[p.cand].type == 'beam' or self.elems[p.anta].type == 'beam':
                    color_id = 3
                elif self.elems[p.cand].type == self.elems[p.anta].type and self.elems[p.anta].type.startswith('stone'):
                    color_id = 4
                elif self.elems[p.cand].type != self.elems[p.anta].type and self.elems[p.anta].type.startswith('stone') and self.elems[p.cand].type.startswith('stone'):
                    color_id = 5
                else:
                    print(
                        f"Unknow contact between {self.elems[p.cand].type} and {self.elems[p.anta].type}!")
            else:
                color_id = p.faceID
            if p.faceID not in contfs.keys():
                color = tab20(color_id % 20)
                contfs[p.faceID] = (color, [p.coor], [p.id])
            else:
                contfs[p.faceID][1].append(p.coor)
                contfs[p.faceID][2].append(p.id)

        _normal_length = 1.0
        for face in contfs.values():
            p_start = face[1][0]
            p_end = face[1][1]
            ax.plot([p_start[0], p_end[0]], [
                p_start[1], p_end[1]], color=face[0], lw=0.3)

            p1_normal_start = (p_start[0]-self.contps[face[2][0]].normal[0]
                               * _normal_length, p_start[1]-self.contps[face[2][0]].normal[1]*_normal_length)
            p1_normal_end = (p_start[0]+self.contps[face[2][0]].normal[0]
                             * _normal_length, p_start[1]+self.contps[face[2][0]].normal[1]*_normal_length)
            ax.plot([p1_normal_start[0], p1_normal_end[0]],
                    [p1_normal_start[1], p1_normal_end[1]], lw=0.1, color=face[0])

            p2_normal_start = (p_end[0]-self.contps[face[2][1]].normal[0]
                               * _normal_length, p_end[1]-self.contps[face[2][1]].normal[1]*_normal_length)
            p2_normal_end = (p_end[0]+self.contps[face[2][1]].normal[0]
                             * _normal_length, p_end[1]+self.contps[face[2][1]].normal[1]*_normal_length)
            ax.plot([p2_normal_start[0], p2_normal_end[0]],
                    [p2_normal_start[1], p2_normal_end[1]], lw=0.1, color=face[0])

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        if invert_y:
            plt.gca().invert_yaxis()
        ax.set_aspect('equal')
        ax.set_axis_off()
        # add a text at the top center of the plot
        plt.text(0.6, 0.95, f'{len(self.elems.keys())} elements, {len(self.contps.keys())} contact points',
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        plt.savefig(filename+'.png', format='png',
                    dpi=600, bbox_inches='tight')

        plt.close()

    def plot_crack(self, filename='crack', displaced=True, disp_factor=1.0):
        # highlight faces whose displacement is different from the counterface
        # assemble contact faces
        max_point_crack_width = -np.inf
        contfs = dict()
        for p in self.contps.values():
            if p.faceID not in contfs.keys():
                displacement_difference = []
                displacement_difference.append(np.linalg.norm(np.asarray(
                    p.displacement)-np.asarray(self.contps[p.counterPoint].displacement)))
                max_point_crack_width = max(
                    max_point_crack_width, np.linalg.norm(np.asarray(p.displacement)-np.asarray(self.contps[p.counterPoint].displacement)))

                if displaced:
                    center = self.elems[p.cand].center
                    node_x = p.coor[0]-center[0]
                    node_y = p.coor[1]-center[1]
                    rot = self.elems[p.cand].displacement[2]*disp_factor
                    trans_x = self.elems[p.cand].displacement[0]*disp_factor
                    trans_y = self.elems[p.cand].displacement[1]*disp_factor

                    new_x = node_x*math.cos(rot)+node_y * \
                        math.sin(rot)+trans_x+center[0]
                    new_y = -node_x*math.sin(rot)+node_y * \
                        math.cos(rot)+trans_y+center[1]

                    contfs[p.faceID] = (
                        displacement_difference, [[new_x, new_y]])
                else:
                    contfs[p.faceID] = (displacement_difference, [p.coor])
            else:
                contfs[p.faceID][0].append(np.linalg.norm(np.asarray(
                    p.displacement)-np.asarray(self.contps[p.counterPoint].displacement)))
                if displaced:
                    center = self.elems[p.cand].center
                    node_x = p.coor[0]-center[0]
                    node_y = p.coor[1]-center[1]
                    rot = self.elems[p.cand].displacement[2]*disp_factor
                    trans_x = self.elems[p.cand].displacement[0]*disp_factor
                    trans_y = self.elems[p.cand].displacement[1]*disp_factor

                    new_x = node_x*math.cos(rot)+node_y * \
                        math.sin(rot)+trans_x+center[0]
                    new_y = -node_x*math.sin(rot)+node_y * \
                        math.cos(rot)+trans_y+center[1]

                    contfs[p.faceID][1].append([new_x, new_y])
                else:
                    contfs[p.faceID][1].append(p.coor)
                max_point_crack_width = max(
                    max_point_crack_width, np.linalg.norm(np.asarray(p.displacement)-np.asarray(self.contps[p.counterPoint].displacement)))

        # plot crack map
        # get the seismic color map
        seismic = plt.get_cmap('Reds', 100)
        fig, ax = plt.subplots()
        for key, value in contfs.items():
            start_point = value[1][0]
            end_point = value[1][1]
            averaged_face_disp = np.average(value[0])
            ax.plot([start_point[0], end_point[0]],
                    [start_point[1], end_point[1]], color=seismic(
                round(100 * (averaged_face_disp/max_point_crack_width))), lw=0.3)
        if not displaced:
            sm = plt.cm.ScalarMappable(cmap=seismic, norm=plt.Normalize(
                vmin=0, vmax=max_point_crack_width))
            sm._A = []
            cbar = fig.colorbar(sm)
            cbar.set_label(f'Max crack width is {max_point_crack_width:0.2f}')
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        # flip y axis
        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.axis('off')

        plt.savefig(filename+'.png', format='png', dpi=600)
        plt.close()

    def plot_normal_force_cmap(self, filename="None"):
        """Plot the normal force of the contact points using a colormap
        """
        # get the seismic color map
        seismic = plt.get_cmap('seismic', 100)
        # get the maximum and minimum normal force
        max_normal_force = max(
            [value.normal_force for key, value in self.contps.items()])
        min_normal_force = min(
            [value.normal_force for key, value in self.contps.items()])
        max_abs_normal_force = max(
            [abs(value.normal_force) for key, value in self.contps.items()])

        fig, ax = plt.subplots()
        lines = []
        color_lines = []
        for key, value in self.contps.items():
            start_point = (value.coor[0]-value.normal[0]
                           * 0.7, value.coor[1]-value.normal[1]*0.7)
            end_point = (value.coor[0]+value.normal[0]
                         * 0.7, value.coor[1]+value.normal[1]*0.7)
            # line = [start_point, end_point]
            # line_color = seismic(
            #    round(50 * ((value.normal_force/max_abs_normal_force)+1)))
            # lines.append(line)
            # color_lines.append(line_color)
            ax.plot([start_point[0], end_point[0]],
                    [start_point[1], end_point[1]], color=seismic(
                round(50 * ((value.normal_force/max_abs_normal_force)+1))), lw=0.3)
            # ax.scatter(value.coor[0], value.coor[1], color=seismic(round(50 *
            #            ((value.normal_force/max_abs_normal_force)+1))), s=2.0)

        # lc = mc.LineCollection(lines, colors=color_lines, linestyle='solid')
        # ax.add_collection(lc)
        # show color bar
        sm = plt.cm.ScalarMappable(cmap=seismic, norm=plt.Normalize(
            vmin=-max_abs_normal_force, vmax=max_abs_normal_force))
        sm._A = []
        cbar = fig.colorbar(sm)
        cbar.set_label(f'Max normal force {max_abs_normal_force:0.2f}')
        plt.axis('off')
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        # flip y axis
        plt.gca().invert_yaxis()

        plt.savefig(filename+'.svg', format='svg')
        plt.close()

    def plot_tangent_force_cmap(self, filename="None"):
        """Plot the normal force of the contact points using a colormap
        """
        # get the seismic color map
        seismic = plt.get_cmap('seismic', 100)
        # get the maximum and minimum normal force
        max_tangent_force = max(
            [value.tangent_force for key, value in self.contps.items()])
        min_tangent_force = min(
            [value.tangent_force for key, value in self.contps.items()])
        max_abs_tangent_force = max(
            [abs(value.tangent_force) for key, value in self.contps.items()])

        fig, ax = plt.subplots()
        for key, value in self.contps.items():
            start_point = (value.coor[0]-value.tangent1[0]
                           * 0.7, value.coor[1]-value.tangent1[1]*0.7)
            end_point = (value.coor[0]+value.tangent1[0]
                         * 0.7, value.coor[1]+value.tangent1[1]*0.7)
            ax.plot([start_point[0], end_point[0]],
                    [start_point[1], end_point[1]], color=seismic(
                round(100 * (abs(value.tangent_force/max_abs_tangent_force)))), lw=0.3)
            # ax.scatter(value.coor[0], value.coor[1], color=seismic(round(50 *
            #            ((value.tangent_force/max_abs_tangent_force)+1))), s=2.0)
        # show color bar
        sm = plt.cm.ScalarMappable(cmap=seismic, norm=plt.Normalize(
            vmin=0, vmax=max_abs_tangent_force))
        sm._A = []
        cbar = fig.colorbar(sm)
        cbar.set_label(f'Max tangent force {max_abs_tangent_force:0.2f}')
        plt.axis('off')
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.gca().invert_yaxis()

        plt.savefig(filename+'.svg', format='svg')
        plt.close()

    def plot_displaced(self, factor=1, save_fig=False, show_fig=True, filename='displace_elements', control_point=0, plot_crack=True, plot_contps=True, plot_element_center=True, invert_y=False):
        """Plot displaced elements and contact points

        :param factor: Amplification of the plotted displacement, defaults to 1
        :type factor: int, optional
        """
        seismic = plt.get_cmap('seismic', 100)

        lines = []
        d = 0
        for key, value in self.elems.items():
            boundary_points = []
            center = value.center
            trans_x = value.displacement[0]*factor
            trans_y = value.displacement[1]*factor
            rot = value.displacement[2]*factor

            for pt in value.vertices:
                node_x = pt[0]-center[0]
                node_y = pt[1]-center[1]

                new_x = node_x*math.cos(rot)+node_y * \
                    math.sin(rot)+trans_x+center[0]
                new_y = -node_x*math.sin(rot)+node_y * \
                    math.cos(rot)+trans_y+center[1]
                boundary_points.append((new_x, new_y))
                # boundary_points.append((p[0], p[1]))

            for i in range(len(boundary_points)):
                lines.append([boundary_points[i-1], boundary_points[i]])
            d += 1
        lc = mc.LineCollection(lines, linewidths=0.3)
        fig, ax = plt.subplots()
        # points
        MC_limit_points_x = []
        MC_limit_points_y = []
        NM_limit_points_x = []
        NM_limit_points_y = []
        MC_NM_limit_points_x = []
        MC_NM_limit_points_y = []
        not_at_limit_points_x = []
        not_at_limit_points_y = []

        #! do not amplify stored displacement directly=>missing amplification in rotation
        if plot_contps:
            for k, value in self.contps.items():
                elem_disp = np.asarray(
                    self.elems[value.cand].displacement)*factor
                trans_x = elem_disp[0]
                trans_y = elem_disp[1]
                rot = elem_disp[2]
                # print(f"element displacement {elem_disp}")
                elem_center = self.elems[value.cand].center
                # print(f"element center {elem_center}")
                node_x = value.coor[0]-elem_center[0]
                node_y = value.coor[1]-elem_center[1]
                new_x = node_x*math.cos(rot)+node_y * \
                    math.sin(rot)+trans_x+elem_center[0]
                new_y = - node_x*math.sin(rot)+node_y * \
                    math.cos(rot)+trans_y+elem_center[1]
                if plot_crack:
                    if value.sliding_failure == True and value.strength_failure == False:
                        MC_limit_points_x.append(new_x)
                        MC_limit_points_y.append(new_y)

                    elif value.sliding_failure == False and value.strength_failure == True:
                        NM_limit_points_x.append(new_x)
                        NM_limit_points_y.append(new_y)

                    elif value.sliding_failure == True and value.strength_failure == True:
                        MC_NM_limit_points_x.append(new_x)
                        MC_NM_limit_points_y.append(new_y)

                    elif value.sliding_failure == False and value.strength_failure == False:
                        not_at_limit_points_x.append(new_x)
                        not_at_limit_points_y.append(new_y)
                    else:
                        # raise error
                        pass
                    # show legend
                    # ax.legend()
                else:
                    color = 'r'
                    # ax.scatter(new_x,
                    #           new_y, c=color)
                    start_point = (new_x-value.normal[0]
                                   * 0.3, new_y-value.normal[1]*0.3)
                    end_point = (new_x+value.normal[0]
                                 * 0.3, new_y+value.normal[1]*0.3)
                    ax.plot([start_point[0], end_point[0]],
                            [start_point[1], end_point[1]], lw=0.5, color=seismic(value.faceID % 5))
                if value.id == control_point:
                    color = 'm'
                    ax.scatter(new_x,
                               new_y, c=color, marker='*', s=50)
                    ax.text(new_x, new_y, f"P{value.id}")
        ax.add_collection(lc)
        if plot_crack:
            ax.scatter(not_at_limit_points_x, not_at_limit_points_y,
                       c='k', label='not at limit', marker='^', alpha=0.3)
            ax.scatter(MC_limit_points_x, MC_limit_points_y, c='r',
                       label='MC limit', marker='<', alpha=0.3)
            ax.scatter(NM_limit_points_x, NM_limit_points_y, c='b',
                       label='NM limit', marker='v', alpha=0.3)
            ax.scatter(MC_NM_limit_points_x, MC_NM_limit_points_y,
                       c='y', label='MC and NM limit', marker='>', alpha=0.3)

            ax.legend()

        # element center
        plot_element_center = True
        if plot_element_center:
            for key, value in self.elems.items():
                if value.type.startswith('stone'):
                    _center_color = 'r'
                elif value.type == 'mortar':
                    _center_color = 'g'
                else:
                    _center_color = 'k'
                ax.scatter(value.center[0]+value.displacement[0]*factor,
                           value.center[1]+value.displacement[1]*factor, s=0.2, c=_center_color)
        plt.axis('equal')
        #plt.xlim(self.xlim)
        #plt.ylim(self.ylim)
        if invert_y:
            plt.gca().invert_yaxis()
        if save_fig:
            plt.axis('off')
            plt.savefig(filename+'.png', format='png', dpi=600)
        if show_fig:
            plt.show()
        plt.close()

    def plot_displaced_img(self, factor=1, img_scale=1, img_id=None, img_type=None, save_fig=True, filename='moved_image', show_fig=False, invert_y=True,):
        plt.imshow(img_type+1, cmap='Greys', vmin=0, vmax=np.max(img_type+1))
        plt.savefig(filename+'_initial.png', dpi=300)
        # plt.imshow(img_id)
        # plt.show()
        moved_image = np.zeros(img_id.shape)
        for key, element in self.elems.items():
            stone_pixels = np.where(img_id == element.id, img_type+1, 0)
            if np.argwhere(stone_pixels != 0).shape[0] == 0:
                print(f"Element {element.id} not found!")
            # plt.imshow(stone_pixels)
            # plt.show()
            rot_center = [element.center[0] *
                          img_scale, element.center[1]*img_scale]
            rot_mat = cv2.getRotationMatrix2D(
                (rot_center[0], rot_center[1]), element.displacement[2]*180/np.pi*factor, 1.0)
            rotated_stone_pixels = cv2.warpAffine(
                stone_pixels, rot_mat, stone_pixels.shape[1::-1], flags=cv2.INTER_NEAREST)
            T = np.float32([[1, 0, element.displacement[0]*factor*img_scale],
                            [0, 1, element.displacement[1]*factor*img_scale]])
            translated_stone_pixels = cv2.warpAffine(
                rotated_stone_pixels, T, img_id.shape[1::-1], flags=cv2.INTER_NEAREST)
            moved_image = np.where((moved_image == 0) & (
                translated_stone_pixels != 0), translated_stone_pixels, moved_image)
            # moved_image += translated_stone_pixels
        plt.clf()
        plt.imshow(moved_image, cmap='Greys', vmin=0, vmax=np.max(img_type+1))
        plt.axis('off')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        if invert_y:
            plt.invert_yaxis()
        if save_fig:
            plt.savefig(filename+'.png', dpi=300)
        if show_fig:
            plt.show()

    def plot_element(self, title='initial elements'):
        lines = []
        d = 0
        for key, value in self.elems.items():
            boundary_points = []
            for p in value.vertices:
                boundary_points.append((p[0], p[1]))

            for i in range(len(boundary_points)):
                lines.append([boundary_points[i-1], boundary_points[i]])
            d += 1
        lc = mc.LineCollection(lines, linewidths=2)
        fig, ax = plt.subplots()

        ax.add_collection(lc)

        for key, value in self.elems.items():
            ax.scatter(value.center[0], value.center[1])
        ax.set_title(title)
        plt.show()

    def get_horizontal_section_force(self, section_y=0, moment_p_x=0.5, section_normal=[0, 1], tolerance=1e-5, plot_fig=True, filename='horizontal_section_force_moment'):
        section_x = []
        tangent_force = []
        normal_force = []
        section_moment = 0
        section_force = 0
        for key, value in self.contps.items():
            if math.isclose(value.coor[1], section_y, abs_tol=tolerance) and (np.allclose(value.normal, section_normal, atol=tolerance) or np.allclose(value.normal, [section_normal[0]*-1, section_normal[1]*-1], atol=tolerance)):
                section_x.append(value.coor[0])
                tangent_force.append(value.tangent_force*value.tangent1[0])
                normal_force.append(value.normal_force)
                section_moment += value.normal_force*(value.coor[0]-moment_p_x)
                section_force += value.normal_force
        if section_x == []:
            print(f'No contact point found at section y = {section_y}!')
            return {}
        if plot_fig:
            # sort list
            section_x, tangent_force, normal_force = zip(
                *sorted(zip(section_x, tangent_force, normal_force)))
            fig, ax = plt.subplots(2, 1)
            line = ax[0].scatter(section_x, tangent_force,
                                 label='contact tangent force')
            line = ax[1].scatter(section_x, normal_force,
                                 label='contact normal force')
            # plt.title(
            #    f'Section at y = {section_y:.2f} and normal direction: {section_normal}\nMoment resp. to x = 0: {section_moment:.2f}\nNormal Force {section_force:.2f}')
            ax[1].set_xlabel('Coordinate x')
            ax[0].set_ylabel('Tangent Force')
            ax[1].set_ylabel('Normal Force')
            plt.savefig(filename+f'.svg', dpi=300)
            plt.close()
        return {"force": round(section_force, int(str(tolerance).split('-')[1])), "moment": section_moment}

    def plot_horizontal_section_force_distribution(self, start_y=0, end_y=1, step=0.1, moment_p_x=0.5, filename='horizontal_section_force_moment_distribution', invert_y=True):
        axial_forces = []
        moments = []
        section_heights = np.arange(start_y, end_y, step).tolist()
        empty_points_height_index = []
        for i_y, section_y in enumerate(section_heights):
            forces = self.get_horizontal_section_force(
                section_y=section_y, moment_p_x=moment_p_x, plot_fig=True, filename=filename+f'_{section_y}')
            if forces == {}:
                empty_points_height_index.append(i_y)
                continue
            axial_forces.append(forces['force'])
            moments.append(forces['moment'])
        if empty_points_height_index != []:
            for index in sorted(empty_points_height_index, reverse=True):
                del section_heights[index]
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(axial_forces, section_heights)
        ax[0].set_title('Axial Force')
        ax[0].set_xlabel('Force')
        ax[0].set_ylabel('Coordinate y')
        ax[1].plot(moments, section_heights)
        ax[1].set_title('Moment')
        ax[1].set_xlabel('Moment')
        if invert_y:
            ax[0].invert_yaxis()
            ax[1].invert_yaxis()
        plt.savefig(filename+'.svg', dpi=300)
