"""
Adapted from "https://github.com/EleMisi/VAEL/blob/main/utils/mario_utils/create_mario_dataset.py"

"""

import os
from itertools import product

import numpy as np
from PIL import Image
from matplotlib import use
from numpy import random
from tqdm import tqdm


# Disable canvas visualization
use('Agg')

ICONS = {
    'glass': f'./mario_icons/glass.png',
    'flowers1': f'./mario_icons/flowers1.png',
    'flowers2': f'./mario_icons/flowers2.png',
    'brick': f'./mario_icons/brick.png',
    'brick2': f'./mario_icons/brick2.png',
    'brick3': f'./mario_icons/brick3.png',
    'concrete': f'./mario_icons/concrete.png',
    'wood': f'./mario_icons/wood.png',
    'white_panel': f'./mario_icons/hite_panel.png',
    'green_panel': f'./mario_icons/green_panel.png',

    'lava': f'./mario_icons/lava.png',
    'sea': f'./mario_icons/sea.png',
    'sand': f'./mario_icons/sand.png',
    'grass': f'./mario_icons/grass.png',
    'chessboard': f'./mario_icons/chessboard.png',
    'chessboard_blue': f'./mario_icons/chessboard_blue.png',
    'chessboard_pink': f'./mario_icons/chessboard_pink.png',
    'brindle': f'./mario_icons/brindle.png',

    'mario': f'./mario_icons/mario.png',
    'luigi': f'./mario_icons/luigi.png',
    'peach': f'./mario_icons/peach.png',
    'bomb': f'./mario_icons/bomb.png',
    'goomba': f'./mario_icons/goomba.png',

    'green_mushroom': f'./mario_icons/green_mushroom.png',
    'star': f'./mario_icons/star.png',
    'red_mushroom': f'./mario_icons/red_mushroom.png',
    'coin': f'./mario_icons/coin.png',
    'cloud': f'./mario_icons/cloud.png'
}

def resize_with_transparency(img, size):
    pal = img.getpalette()
    width, height = img.size
    actual_transp = img.info['actual_transparency']  # XXX This will fail.

    result = Image.new('LA', img.size)

    im = img.load()
    res = result.load()
    for x in range(width):
        for y in range(height):
            t = actual_transp[im[x, y]]
            color = pal[im[x, y]]
            res[x, y] = (color, t)

    return result.resize(size, Image.ANTIALIAS)


def PNG_ResizeKeepTransparency(img, new_width=0, new_height=0, resample="LANCZOS", RefFile=''):
    # needs PIL
    # Inputs:
    #   - SourceFile  = initial PNG file (including the path)
    #   - ResizedFile = resized PNG file (including the path)
    #   - new_width   = resized width in pixels; if you need % plz include it here: [your%] *initial width
    #   - new_height  = resized hight in pixels ; default = 0 = it will be calculated using new_width
    #   - resample = "NEAREST", "BILINEAR", "BICUBIC" and "ANTIALIAS"; default = "ANTIALIAS"
    #   - RefFile  = reference file to get the size for resize; default = ''

    img = img.convert("RGBA")  # convert to RGBA channels
    width, height = img.size  # get initial size

    # if there is a reference file to get the new size
    if RefFile != '':
        imgRef = Image.open(RefFile)
        new_width, new_height = imgRef.size
    else:
        # if we use only the new_width to resize in proportion the new_height
        # if you want % of resize please use it into new_width (?% * initial width)
        if new_height == 0:
            new_height = new_width * width / height

    # split image by channels (bands) and resize by channels
    img.load()
    bands = img.split()
    # resample mode
    if resample == "NEAREST":
        resample = Image.NEAREST
    else:
        if resample == "BILINEAR":
            resample = Image.BILINEAR
        else:
            if resample == "BICUBIC":
                resample = Image.BICUBIC
            else:
                if resample == "ANTIALIAS":
                    resample = Image.ANTIALIAS
                else:
                    if resample == "LANCZOS":
                        resample = Image.LANCZOS
    bands = [b.resize((new_width, new_height), resample) for b in bands]
    # merge the channels after individual resize
    img = Image.merge('RGBA', bands)

    return img


def draw_mario_world(X, Y, agent_x, agent_y, target_x, target_y, agent_icon='goomba', target_icon='green_mushroom',
                     background_tile='lava', frame_tile='glass'):
    """
    This method creates the specified Mario's world.
    """
    # Initialize canvas
    W, H = 20, 20
    image = Image.new("RGBA", ((X + 2) * W, (Y + 2) * H), (255, 255, 255))
    # Define y offset for PIL
    agent_y = - (agent_y - (Y - 1))
    target_y = - (target_y - (Y - 1))
    # Set off-set due to frame_dict
    agent_x, agent_y = agent_x + 1, agent_y + 1
    target_x, target_y = target_x + 1, target_y + 1
    # Scale position to tile dimension
    agent_x, agent_y = agent_x * W, agent_y * H
    target_x, target_y = target_x * W, target_y * H
    # Load mario_icons and tiles
    agent_icon = Image.open(ICONS[agent_icon])
    target_icon = Image.open(ICONS[target_icon])
    background_tile = Image.open(ICONS[background_tile])
    frame_tile = Image.open(ICONS[frame_tile])
    # Resize mario_icons and tiles to fit the image

    background_tile = background_tile.resize((W, H), Image.LANCZOS)
    frame_tile = frame_tile.resize((W, H), Image.LANCZOS)
    agent_icon = PNG_ResizeKeepTransparency(agent_icon, new_width=int(W / 2), new_height=int(H / 2), resample="LANCZOS",
                                            RefFile='')
    target_icon = PNG_ResizeKeepTransparency(target_icon, new_width=int(W / 2) + 2, new_height=int(H / 2) + 2,
                                             resample="LANCZOS",
                                             RefFile='')
    # Define frame_dict tiles left corners
    frame_tiles_pos = []
    for i in range(Y + 2):
        frame_tiles_pos.append((0, i * H))
        frame_tiles_pos.append(((X+1) * W, i * H))
        frame_tiles_pos.append((i * W, 0))
        frame_tiles_pos.append((i * W, (X+1) * H))
    # Define background_dict tiles left corners
    bkg_tiles_pos = []
    for i in range(1, Y + 1):
        for j in range(1, X + 1):
            bkg_tiles_pos.append((j * W, i * H))
    # Draw frame_dict
    for box in frame_tiles_pos:
        image.paste(frame_tile, box=box)
    # Draw background_dict
    for box in bkg_tiles_pos:
        image.paste(background_tile, box=box)
    # Draw target_dict
    # target_box = (target_x + 4, target_y + 4)
    # image.paste(target_icon, box=target_box, mask=target_icon)
    # Draw agent_dict
    agent_box = (agent_x + 5, agent_y + 5)
    image.paste(agent_icon, box=agent_box, mask=agent_icon)

    return np.array(image)[:, :, :3]

def define_program(traj):
    """
    Translate the given trajectory in Mario program

    traj: list containing pairs of sequentially 2D Mario coordinates [((x0,y0),(x1,y1)), ((x1,y1),(x2,y2)),...]
    """
    program = []
    # Tras
    for (x0, y0), (x1, y1) in traj:
        if x0 < x1:
            program.append("right")
        elif x0 > x1:
            program.append("left")
        elif y0 < y1:
            program.append("up")
        elif y0 > y1:
            program.append("down")

    return program

def create_mario_dataset(folder):

    # List of 9 pairs of agent positions in a 3x3 grid
    position_set = set()
    for x_start, y_start in product([0], [0, 1, 2]):
        for x_finish, y_finish in [(2,2),(1,2)]:
            if x_finish - x_start < 0 or y_finish - y_start < 0:
                continue
            if x_finish - x_start == 0 and y_finish - y_start == 0:
                continue
            pos = list()
            x_now = x_start
            y_now = y_start
            pos.append((x_now, y_now))
            while x_finish - x_now > 0:
                x_now = x_now + 1
                pos.append((x_now, y_now))
            while y_finish - y_now > 0:
                y_now = y_now + 1
                pos.append((x_now, y_now))
            pos = tuple(pos)
            if len(pos) > 2:
                position_set.add(pos)

    # Order positions list for reproducibility
    position_list = list(position_set)
    position_list.sort(key=lambda y: (y[0], y[1]))

    positions_move = []
    position = []
    move = []
    for pos in position_list:
        p = []
        m = []
        for i in range(len(pos)-1):
            m.append(define_program([(pos[i], pos[i+1])])[0])
            p.append(pos[i])
        p.append(pos[len(pos)-1])
        p_m = p + m
        p_m = tuple(p_m)
        p = tuple(p)
        m = tuple(m)
        positions_move.append(p_m)
        position.append(p)
        move.append(m)

    # Create Dataset from positions and moves
    if not os.path.exists(folder):
        os.makedirs(folder)

    agents = ['peach', 'mario', 'luigi']
    targets = ['coin']
    backgrounds = ['chessboard_blue', 'sea', 'grass', 'chessboard_pink', 'chessboard', 'sand','flowers1','flowers2']
    frames = ['brick', 'brick2', 'brindle', 'brick3', 'glass','concrete','wood']
    images = {'all': []}
    moves = {'all': []}
    positions = {'all': []}
    target_pos = {'all': []}
    agent_dict = {'all': []}
    target_dict = {'all': []}
    background_dict = {'all': []}
    frame_dict = {'all': []}

    info = {
        'agent_dict': {'all': {c: 0 for c in agents}},
        'target_dict': {'all': {o: 0 for o in targets}},
        'background_dict': {'all': {c: 0 for c in backgrounds}},
        'frame_dict': {'all': {o: 0 for o in frames}},
        'pos': {'all': {str(p): 0 for p in position}},
        'moves': {'all': {str(m): 0 for m in move}}}

    configs = list(product(agents, targets, backgrounds, frames))
    # Order config list for reproducibility
    configs.sort()
    idxs_split = {'all': []}
    tot_imgs = len(configs) * len(positions_move)

    idxs = list(range(tot_imgs))
    random.seed(88888)
    idxs_split['all'] = random.choice(idxs, size=tot_imgs, replace=False)

    idx = 0
    for config in tqdm(configs):
        (a, t, bg, f) = config
        for i in range(len(position)):
            p = position[i]
            p_list = list(p)
            m = move[i]
            m_list = list(m)
            imgs = []
            imgs_num = 0
            for pos in p:
                img = draw_mario_world(X=3, Y=3, agent_x=pos[0], agent_y=pos[1], target_x=2, target_y=2, agent_icon=a, target_icon=t, background_tile=bg, frame_tile=f)
                imgs.append(img)
                imgs_num = imgs_num + 1
            while imgs_num < 5:
                img = draw_mario_world(X=3, Y=3, agent_x=pos[0], agent_y=pos[1], target_x=2, target_y=2, agent_icon=a,
                                       target_icon=t, background_tile=bg, frame_tile=f)
                imgs.append(img)
                imgs_num = imgs_num + 1
            for dataset in ['all']:
                if idx in idxs_split[dataset]:
                    images[dataset].append(imgs)
                    moves[dataset].append(m_list)
                    positions[dataset].append(p_list)
                    target_pos[dataset].append(pos)
                    agent_dict[dataset].append(a)
                    target_dict[dataset].append(t)
                    background_dict[dataset].append(bg)
                    frame_dict[dataset].append(f)
                    info['target_dict'][dataset][t] += 1
                    info['frame_dict'][dataset][f] += 1
                    info['background_dict'][dataset][bg] += 1
                    info['agent_dict'][dataset][a] += 1
                    info['pos'][dataset][str(p)] += 1
                    info['moves'][dataset][str(m)] += 1
            idx += 1
    # Check dimensions
    assert len(images['all']) == tot_imgs

    # Save images, moves and positions
    for dataset in ['all']:
        np.savez(os.path.join(folder, f'mario_pos.npz'), images=images[dataset],
                 pos=positions[dataset], target_pos=target_pos[dataset], moves=moves[dataset], agents=agent_dict[dataset], targets=target_dict[dataset],
                 bkgs=background_dict[dataset], frames=frame_dict[dataset])

if __name__ == '__main__':
    create_mario_dataset('/home/worker/exp/AbdGen/dataset/new')
