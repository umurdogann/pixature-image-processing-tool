# region IMPORTED LIBRARIES
import tkinter as tk
from tkinter import DoubleVar, ttk
import Operations as op
from tkinter import messagebox
import time
# endregion

# region Events


def ClearContainer():
    for target_list in opPanel.winfo_children():
        target_list.pack_forget()
        pass
    try:
        op.opImg = op.copy.deepcopy(op.cvImg)
        op.DisplayImage(op.opImg)
    except:
        pass

    pass


def btnContrast_ClickEvent():
    ClearContainer()
    pnlContrast.pack(fill=tk.BOTH, expand=True)
    lblOpName.config(text='CONTRAST')
    pass


def btnFileDialog_ClickEvent():
    try:
        ClearContainer()
    except IndexError as identifier:
        pass
    op.panel = containerPanel
    if(op.LoadImage() == True):
        PackLeftPanel()
        master.title('Pixature @'+op.imagePath)
        pass
    pass


def tbContrast_ValueChangedEvent(alpha):
    lblAlpha.config(text='Alpha Value : ' + alpha)
    f_alpha = float(alpha)
    op.Contrast(f_alpha)
    pass


def btnContrastReset_ClickEvent():
    tbContrast.set(1.0)
    op.ResetImage()
    pass


def btnContrastSave_ClickEvent():
    op.SaveImage()
    tbContrast.set(1.0)
    pass


def btnRotate_ClickEvent():
    ClearContainer()
    pnlRotate.pack(fill=tk.BOTH, expand=True)
    lblOpName.config(text='ROTATE')
    tbRotate.set(0)
    pass


def tbRotate_ValueChangedEvent(degree):
    lblDegree.config(text='Degree : ' + degree)
    i_degree = int(degree)
    op.Rotate(i_degree)
    pass


def btnRotateSave_ClickEvent():
    op.SaveImage()
    tbRotate.set(0)
    pass


def btnRotateReset_ClickEvent():
    tbRotate.set(0)
    op.ResetImage()
    pass


def btnMirror_ClickEvent():
    ClearContainer()
    pnlMirror.pack(fill=tk.BOTH, expand=True)
    lblOpName.config(text='MIRROR')
    pass


def btnMirrorSave_ClickEvent():
    op.SaveImage()
    pass


def btnCrop_ClickEvent():
    ClearContainer()
    pnlCrop.pack(fill=tk.BOTH, expand=True)
    lblOpName.config(text='CROP')
    tbCrop.set(0)
    op.CropInit()
    pass


def btnCropSave_ClickEvent():
    op.SaveImage()
    op.ResetImage()
    op.CropTempSave()
    tbCrop.set(0)
    pass


def btnCropReset_ClickEvent():
    op.ResetImage()
    op.CropTempSave()
    tbCrop.set(0)
    pass


def tbCrop_ValueChangedEvent(perc):
    op.Crop(rbCropValue.get(), perc)
    pass


def rbCrop_ClickEvent():
    op.CropTempSave()
    tbCrop.set(0)


def btnBns_ClickEvent():
    ClearContainer()
    pnlBns.pack(fill=tk.BOTH, expand=True)
    lblOpName.config(text='BRIGHTNESS/DARKNESS')
    tbBns.set(0)
    pass


def tbBns_ValueChangedEvent(val):
    i_val = int(val)
    op.Brightness(i_val)
    lblBnsValue.config(text='Beta Value : ' + val)
    pass


def btnBnsReset_ClickEvent():
    tbBns.set(0)
    op.ResetImage()
    pass


def btnBnsSave_ClickEvent():
    op.SaveImage()
    tbBns.set(0)
    pass


def btnBlur_ClickEvent():
    ClearContainer()
    pnlBlur.pack(fill=tk.BOTH, expand=True)
    lblOpName.config(text='BLUR')
    tbBlurX.set(1)
    tbBlurY.set(1)
    pass


def tbBlurX_ValueChangedEvent(kernel_x):
    op.Blur(rbBlurValue.get(), int(kernel_x), int(tbBlurY.get()))
    pass


def tbBlurY_ValueChangedEvent(kernel_y):
    op.Blur(rbBlurValue.get(), int(tbBlurX.get()), int(kernel_y))
    pass


def rbBlur_ClickEvent():
    op.DisplayImage(op.opImg)
    tbBlurX.set(0)
    tbBlurY.set(0)
    pass


def btnBlurSave_ClickEvent():
    op.SaveImage()
    tbBlurX.set(0)
    tbBlurY.set(0)
    pass


def btnBlurReset_ClickEvent():
    op.ResetImage()
    tbBlurX.set(0)
    tbBlurY.set(0)
    pass


def btnInvert_ClickEvent():
    ClearContainer()
    pnlInvert.pack(fill=tk.BOTH, expand=True)
    lblOpName.config(text='INVERT')
    pass


def btnDoInvert_ClickEvent():
    op.Invert()
    pass


def btnInvertReset_ClickEvent():
    op.ResetImage()
    pass


def btnInvertSave_ClickEvent():
    op.SaveImage()
    pass


def btnMorp_ClickEvent():
    ClearContainer()
    pnlMorp.pack(fill=tk.BOTH, expand=True)
    lblOpName.config(text='MORPHOLOGICAL TRANSFORMATION')
    tbMorpX.set(0)
    tbMorpY.set(0)
    pass


def tbMorpX_ValueChangedEvent(kernel_x):
    op.MorphoTransform(rbMorpValue.get(), int(kernel_x), int(tbMorpY.get()))
    pass


def tbMorpY_ValueChangedEvent(kernel_y):
    op.MorphoTransform(rbMorpValue.get(), int(tbMorpX.get()), int(kernel_y))
    pass


def rbMorp_ClickEvent():
    op.opImg = op.copy.deepcopy(op.cvImg)
    op.DisplayImage(op.opImg)
    tbMorpX.set(0)
    tbMorpY.set(0)
    pass


def btnMorpSave_ClickEvent():
    op.SaveImage()
    tbMorpX.set(0)
    tbMorpY.set(0)
    pass


def btnMorpReset_ClickEvent():
    op.ResetImage()
    tbMorpX.set(0)
    tbMorpY.set(0)
    pass


def btnHistNorm_ClickEvent():
    ClearContainer()
    pnlHistNorm.pack(fill=tk.BOTH, expand=True)
    lblOpName.config(text='HISTOGRAM NORMALIZATION')
    tbHistShift.set(0.0)
    tbHistCap.set(1.0)


def btnHistSave_ClickEvent():
    op.SaveImage()
    tbHistCap.set(1.0)
    tbHistShift.set(0.0)
    pass


def btnHistReset_ClickEvent():
    op.ResetImage()
    tbHistCap.set(1.0)
    tbHistShift.set(0.0)
    pass


def tbHistShift_ValueChangedEvent(val):
    f_val = float(val)
    op.HistogramNormalization(f_val, float(tbHistCap.get()))
    pass


def tbHistCap_ValueChangedEvent(val):
    f_val = float(val)
    op.HistogramNormalization(float(tbHistShift.get()), f_val)
    pass


def btnMorp_ClickEvent():
    ClearContainer()
    pnlMorp.pack(fill=tk.BOTH, expand=True)
    lblOpName.config(text='MORPHOLOGICAL TRANSFORMATION')
    pass


def btnColor_ClickEvent():
    ClearContainer()
    pnlColor.pack(fill=tk.BOTH, expand=True)
    lblOpName.config(text='COLOR CHANNELS')
    pass


def btnColorSave_ClickEvent():
    op.SaveImage()
    tbColorR.set(0)
    tbColorG.set(0)
    tbColorB.set(0)
    pass


def btnColorReset_ClickEvent():
    op.ResetImage()
    tbColorR.set(0)
    tbColorG.set(0)
    tbColorB.set(0)
    pass


def tbColorR_ValueChangedEvent(val):
    i_val = int(val)
    g_val = int(tbColorG.get())
    b_val = int(tbColorB.get())
    op.ColorChannels(i_val, g_val, b_val)
    pass


def tbColorG_ValueChangedEvent(val):
    r_val = int(tbColorR.get())
    i_val = int(val)
    b_val = int(tbColorB.get())
    op.ColorChannels(r_val, i_val, b_val)
    pass


def tbColorB_ValueChangedEvent(val):
    r_val = int(tbColorR.get())
    g_val = int(tbColorG.get())
    i_val = int(val)
    op.ColorChannels(r_val, g_val, i_val)
    pass


def btnFilters_ClickEvent():
    ClearContainer()
    pnlFilters.pack(fill=tk.BOTH, expand=True)
    lblOpName.config(text='ARTISTIC FILTERS')
    pass


def btnFilterSave_ClickEvent():
    op.SaveImage()
    pass


def btnFilterReset_ClickEvent():
    op.ResetImage()
    pass

def btnAIE_ClickEvent():
    ClearContainer()
    pnlAIE.pack(fill=tk.BOTH, expand=True)
    lblOpName.config(text='Brightness / Contrast Regulator')
    pass

def btnAutoBrightAndContrast_ClickEvent():
    op.AutomaticBrightnessAndContrast(3)
    pass

def btnAIESave_ClickEvent():
    op.SaveImage()
    pass

def btnAIEReset_ClickEvent():
    op.ResetImage()
    pass

def btnUndo_ClickEvent():
    op.Undo()
    tbRotate.set(0)
    tbContrast.set(1.0)
    pass


def btnRedo_ClickEvent():
    op.Redo()
    tbRotate.set(0)
    tbContrast.set(1.0)
    pass


def PackLeftPanel():  # Packing left panel components after loading image
    btnRotate.pack(fill=tk.X, padx=5, pady=5)
    btnCrop.pack(fill=tk.X, padx=5, pady=5)
    btnInvert.pack(fill=tk.X, padx=5, pady=5)
    btnMirror.pack(fill=tk.X, padx=5, pady=5)
    btnBns.pack(fill=tk.X, padx=5, pady=5)
    btnContrast.pack(fill=tk.X, padx=5, pady=5)
    btnBlur.pack(fill=tk.X, padx=5, pady=5)
    btnHistNorm.pack(fill=tk.X, padx=5, pady=5)
    btnMorp.pack(fill=tk.X, padx=5, pady=5)
    btnColor.pack(fill=tk.X, padx=5, pady=5)
    btnFilters.pack(fill=tk.X, padx=5, pady=5)
    btnAIE.pack(fill=tk.X,padx=5,pady=5)

# endregion

# region FILTERS
# -----------------------------------------FILTERS---------------------------------


def btnDownsideUpFilter_ClickEvent():
    op.downsideUpFilter()
    pass


def btnSoftBWFilter_ClickEvent():
    op.SoftBWfilter()
    pass


def btnCartoonizerEffectFilter_ClickEvent():
    op.cartoonizerEffectFilter()
    pass


def btnAsheFilter_ClickEvent():
    op.asheFilter()
    pass


def btnBRossFilter_ClickEvent():
    op.BRossFilter()
    pass


def btnCoolFilter_ClickEvent():
    op.coolFilter()
    pass


def btnNegativeFilter_ClickEvent():
    op.negativeFilter()
    pass


def btnCarbonPaperFilter_ClickEvent():
    op.carbonPaperFilter()
    pass


def btnWarmFilter_ClickEvent():
    op.warmFilter()
    pass


def btnMasterSketch_ClickEvent():
    op.masterSketcherFilter()
    pass


def btnColoredMasterSketch_ClickEvent():
    op.coloredMasterSketcherFilter()
    pass


def btnEmboss_ClickEvent():
    op.embossFilter()
    pass


def btnDownsideNeon_ClickEvent():
    op.downsideNeonFilter()
    pass


def btnMarked_ClickEvent():
    op.markedFilter()
    pass


def btnblackSunny_ClickEvent():
    op.blackSunny()
    pass


def btnvividNeon_ClickEvent():
    op.vividNeon()
    pass


def btnlala_ClickEvent():
    op.lala()
    pass


def btnwonderland_ClickEvent():
    op.wonderland()
    pass


def btnsundown_ClickEvent():
    op.sundown()
    pass


def btnglossy_ClickEvent():
    op.glossy()
    pass


def btnhandDrawn_ClickEvent():
    op.handDrawn()
    pass


def btnoldTown_ClickEvent():
    op.oldTown()
    pass


def btnatDawn_ClickEvent():
    op.atDawn()
    pass


def btncandyGirl_ClickEvent():
    op.candyGirl()
    pass
# endregion

# region MAIN FORM
master = tk.Tk()
rbCropValue = tk.StringVar(master, 'left')
rbBlurValue = tk.StringVar(master, 'gaussian')
rbMorpValue = tk.StringVar(master, 'opening')
master.title('Pixature')
master.geometry('800x600')


# PANEL LEFT
leftPanel = tk.Frame(master, bg='#353535')
leftPanel.pack(fill=tk.Y, side=tk.LEFT)
# PANEL FILEDIALOG
filePanel = tk.Frame(leftPanel, bg='#252525')
filePanel.pack(fill=tk.X)
# LABEL FILEDIALOG
lblFileDialog = tk.Label(filePanel, text='CENG471 Final Project',
                         bg='#252525', fg='#FFFFFF', font=('Courier', 12, 'bold'))
lblFileDialog.pack(fill=tk.X, expand=True, padx=5, pady=6)
# BUTTON ROTATE
btnRotate = tk.Button(leftPanel, text='Rotate', bg='#404040',
                      fg='White', command=btnRotate_ClickEvent)

# BUTTON CROP
btnCrop = tk.Button(leftPanel, text='Crop', bg='#404040',
                    fg='White', command=btnCrop_ClickEvent)

# BUTTON INVERT
btnInvert = tk.Button(leftPanel, text='Invert', bg='#404040',
                      fg='White', command=btnInvert_ClickEvent)

# BUTTON MIRROR
btnMirror = tk.Button(leftPanel, text='Mirror', bg='#404040',
                      fg='White', command=btnMirror_ClickEvent)

# BUTTON BRIGHTNESS/DARKNESS
btnBns = tk.Button(leftPanel, text='Brightness/Darkness', bg='#404040', fg='White',
                   command=btnBns_ClickEvent)

# BUTTON CONTRAST
btnContrast = tk.Button(leftPanel, text='Contrast', bg='#404040', fg='White',
                        command=btnContrast_ClickEvent)

# BUTTON BLUR
btnBlur = tk.Button(leftPanel, text='Blur', bg='#404040', fg='White',
                    command=btnBlur_ClickEvent)

# BUTTON COLOR
btnColor = tk.Button(leftPanel, text='Color Channels', bg='#404040', fg='White',
                     command=btnColor_ClickEvent)

# BUTTON HISTOGRAM NORMALIZATION
btnHistNorm = tk.Button(leftPanel, text='Histogram Normalization', bg='#404040', fg='White',
                        command=btnHistNorm_ClickEvent)

# BUTTON MORPHOLOGICAL TRANSFORMATION
btnMorp = tk.Button(leftPanel, text='Morphological Transformation', bg='#404040', fg='White',
                    command=btnMorp_ClickEvent)

# BUTTON FILTERS
btnFilters = tk.Button(leftPanel, text='Artistic Filters / Effects',
                       bg='#404040', fg='White', command=btnFilters_ClickEvent)

# BUTTON AUTOMATIC IMAGE ENHANCEMENT
btnAIE = tk.Button(leftPanel, text='Brightness / Contrast Regulator',
                       bg='#404040', fg='White', command=btnAIE_ClickEvent)

pnlIcon = tk.Label(leftPanel,bg='#353535')
pnlIcon.pack(fill=tk.X,side=tk.BOTTOM)
imgIcon = op.cv2.imread('pixature-transparent.png')
b, g, r = op.cv2.split(imgIcon)
imgIcon = op.cv2.merge((r,g,b))
im = op.Image.fromarray(imgIcon)
im = im.resize((120, 120), op.Image.ANTIALIAS) 
imgtk = op.ImageTk.PhotoImage(image=im)
pnlIcon.configure(image=imgtk)
pnlIcon.image = imgtk

# PANEL RIGHT
rightPanel = tk.Frame(master, width=500, height=600)
rightPanel.pack(fill=tk.BOTH, expand=True)
# SEPERATOR BETWEEN RIGHT AND LEFT PANELS
seperatorLR = ttk.Separator(rightPanel, orient='vertical')
#seperatorLR.pack(fill=tk.Y, side=tk.LEFT)
# PANEL TOP
topPanel = tk.Frame(rightPanel, bg='#252525')
topPanel.pack(side=tk.TOP, fill=tk.X)
# BUTTON REDO
btnRedo = tk.Button(topPanel, text='REDO', bg='DarkBlue',
                    fg='White', command=btnRedo_ClickEvent)
btnRedo.pack(side=tk.RIGHT, padx=5, pady=5)
# BUTTON UNDO
btnUndo = tk.Button(topPanel, text='UNDO', bg='Maroon',
                    fg='White', command=btnUndo_ClickEvent)
btnUndo.pack(side=tk.RIGHT, padx=5, pady=5)
# LABEL OPERATION NAME
lblOpName = tk.Label(topPanel, text='Operation Name',
                     bg='#252525', fg='#FFFFFF', font=('Courier', 12, 'bold'))
lblOpName.pack(fill=tk.X, padx=5, pady=5)
# SEPERATOR BETWEEN TOP PANEL AND CONTAINER
separatorTC = ttk.Separator(rightPanel, orient='horizontal')
# separatorTC.pack(fill=tk.X)
# PANEL CONTAINER
containerPanel = tk.Label(rightPanel, bg='#303030')
containerPanel.pack(fill=tk.BOTH, expand=True)
# SEPERATOR BETWEEN CONTAINER AND OPARATION PANELS
separatorCO = tk.Frame(rightPanel, height=1, bg='#404040')
separatorCO.pack(fill=tk.X)
# PANEL OPERATION
opPanel = tk.Frame(rightPanel, height=150, bg='#303030')
opPanel.pack(fill=tk.BOTH)
# endregion

# region MENUBAR
# MENUBAR
menubar = tk.Menu(master, background='#454545', foreground='white',
                  activebackground='#757575', activeforeground='white')
filemenu = tk.Menu(menubar, tearoff=0, background='#454545', foreground='white',
                   activebackground='#757575', activeforeground='white')
filemenu.add_command(label="Open an Image", command=btnFileDialog_ClickEvent)
filemenu.add_command(label="Save Image", command=op.SaveImageFile)
filemenu.add_command(label="Save as...", command=op.SaveImageFileAs)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=master.quit)
helpMenu = tk.Menu(menubar, tearoff=0, background='#454545', foreground='white',
                   activebackground='#757575', activeforeground='white')
helpMenu.add_command(label='How to use?', command=lambda: tk.messagebox.showinfo('Info', 'Pixature.\nTo start using this app, you should open an image to the app.\nThen, you can apply some operations such as rotating, cropping, inverting on the image.\nWhen you applied an operation on the image, please don\'t forget to save with the save button on the bottom panel. If you don\'t save the operation, when you click another operation button, your changes would be lost. Also, if you don\'t save the image, you cannot use undo and redo features.\n To save all changes as a .png, you should open the menubar where is the top of the app and click on the \'Save Image\' or \' Save as...\' buttons.'))
helpMenu.add_command(label='About us...', command=lambda: tk.messagebox.showinfo(
    'About Us', 'Team Member: Beyza AKKOYUN\nTeam Member: Eda Nur AR\nTeam Member: Cuma Umur DOĞAN\nFor CENG471\n≧◉◡◉≦'))
menubar.add_cascade(label="File", menu=filemenu)
menubar.add_cascade(label='Help', menu=helpMenu)

# endregion

# region COLOR CHANNELS
# PANEL COLOR CHANNELS
pnlColor = tk.Frame(opPanel, bg='#303030')
pnlColor.pack(fill=tk.BOTH, expand=True)
pnlColor.pack_forget()

# TRACK BAR R
tbColorR = tk.Scale(pnlColor, from_=-255, to=255, resolution=1, label='R Channel', bg='#303030', fg='White', highlightthickness=0,
                    orient='horizontal', command=tbColorR_ValueChangedEvent)
tbColorR.set(0)
tbColorR.pack(fill=tk.X, padx=20)

# TRACK BAR G
tbColorG = tk.Scale(pnlColor, from_=-255, to=255, resolution=1, label='G Channel', bg='#303030', fg='White', highlightthickness=0,
                    orient='horizontal', command=tbColorG_ValueChangedEvent)
tbColorG.set(0)
tbColorG.pack(fill=tk.X, padx=20)

# TRACK BAR B
tbColorB = tk.Scale(pnlColor, from_=-255, to=255, resolution=1, label='B Channel', bg='#303030', fg='White', highlightthickness=0,
                    orient='horizontal', command=tbColorB_ValueChangedEvent)
tbColorB.set(0)
tbColorB.pack(fill=tk.X, padx=20)

# BUTTON SAVE
btnColorSave = tk.Button(pnlColor, text='Save', bg='#353535', fg='White',
                         command=btnColorSave_ClickEvent)
btnColorSave.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(5, 20))

# BUTTON RESET
btnColorReset = tk.Button(pnlColor, text='Reset', bg='#353535', fg='White',
                          command=btnColorReset_ClickEvent)
btnColorReset.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(5, 0))

# endregion

# region MORPHOLOGICAL TRANSFORMATION
# PANEL MORPHOLOGICAL TRANSFORMATION
pnlMorp = tk.Frame(opPanel, bg='#303030')
pnlMorp.pack(fill=tk.BOTH, expand=True)
pnlMorp.pack_forget()

# TRACK BAR MORP KX
tbMorpX = tk.Scale(pnlMorp, from_=0, to=100, resolution=1, label='Kernel X', bg='#303030', fg='White', highlightthickness=0,
                   orient='horizontal', command=tbMorpX_ValueChangedEvent)
tbMorpX.set(0)
tbMorpX.pack(fill=tk.X, padx=20)

# TRACK BARMORP KY
tbMorpY = tk.Scale(pnlMorp, from_=0, to=100, resolution=1, label='Kernel Y', bg='#303030', fg='White', highlightthickness=0,
                   orient='horizontal', command=tbMorpY_ValueChangedEvent)
tbMorpY.set(0)
tbMorpY.pack(fill=tk.X, padx=20)

# PANEL RADIOBUTTONS MORP
pnlMorpRB = tk.Frame(pnlMorp, bg='#303030')
pnlMorpRB.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

# RADIOBUTTON MORP OPENING
rbMorpOpening = tk.Radiobutton(
    pnlMorpRB, text='Opening', variable=rbMorpValue, value='opening', bg='#353535', fg='White', activebackground='#353535', activeforeground='White', selectcolor='#353535', command=rbMorp_ClickEvent)
rbMorpOpening.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# RADIOBUTTON MORP CLOSING
rbMorpClosing = tk.Radiobutton(
    pnlMorpRB, text='Closing', variable=rbMorpValue, value='closing', bg='#353535', fg='White', activebackground='#353535', activeforeground='White', selectcolor='#353535', command=rbMorp_ClickEvent)
rbMorpClosing.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# RADIOBUTTON MORP GRADIENT
rbMorpGradient = tk.Radiobutton(
    pnlMorpRB, text='Gradient', variable=rbMorpValue, value='gradient', bg='#353535', fg='White', activebackground='#353535', activeforeground='White', selectcolor='#353535', command=rbMorp_ClickEvent)
rbMorpGradient.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# RADIOBUTTON MORP TOPHAT
rbMorpTophat = tk.Radiobutton(
    pnlMorpRB, text='Tophat', variable=rbMorpValue, value='tophat', bg='#353535', fg='White', activebackground='#353535', activeforeground='White', selectcolor='#353535', command=rbMorp_ClickEvent)
rbMorpTophat.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# RADIOBUTTON MORP BLACKHAT
rbMorpBlackhat = tk.Radiobutton(
    pnlMorpRB, text='Blackhat', variable=rbMorpValue, value='blackhat', bg='#353535', fg='White', activebackground='#353535', activeforeground='White', selectcolor='#353535', command=rbMorp_ClickEvent)
rbMorpBlackhat.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTTON SAVE
btnMorpSave = tk.Button(pnlMorp, text='Save', bg='#353535', fg='White',
                        command=btnMorpSave_ClickEvent)
btnMorpSave.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(5, 20))

# BUTTON RESET
btnMorpReset = tk.Button(pnlMorp, text='Reset', bg='#353535', fg='White',
                         command=btnMorpReset_ClickEvent)
btnMorpReset.pack(fill=tk.X, side=tk.BOTTOM, padx=20)

# endregion

# region HISTOGRAM NORMALIZATION
# PANEL HISTNOR
pnlHistNorm = tk.Frame(opPanel, bg='#303030')
pnlHistNorm.pack(fill=tk.BOTH, expand=True)
pnlHistNorm.pack_forget()

# TRACK BAR HISTNOR SHIFT
tbHistShift = tk.Scale(pnlHistNorm, from_=-2, to=2, resolution=0.1, label='Shift', bg='#303030', fg='White', highlightthickness=0,
                       orient='horizontal', command=tbHistShift_ValueChangedEvent)
tbHistShift.set(0)
tbHistShift.pack(fill=tk.X, padx=20)

# TRACK BAR HISTNOR CAP
tbHistCap = tk.Scale(pnlHistNorm, from_=-2, to=2, resolution=0.1, label='Cap', bg='#303030', fg='White', highlightthickness=0,
                     orient='horizontal', command=tbHistCap_ValueChangedEvent)
tbHistCap.set(1.0)
tbHistCap.pack(fill=tk.X, padx=20, pady=(0, 5))

# BUTTON SAVE
btnHistSave = tk.Button(pnlHistNorm, text='Save', bg='#353535', fg='White',
                        command=btnHistSave_ClickEvent)
btnHistSave.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(5, 20))

# BUTTON RESET
btnHistReset = tk.Button(pnlHistNorm, text='Reset', bg='#353535', fg='White',
                         command=btnHistReset_ClickEvent)
btnHistReset.pack(fill=tk.X, side=tk.BOTTOM, padx=20)
# endregion

# region BLUR
# PANEL BLUR
pnlBlur = tk.Frame(opPanel, bg='#303030')
pnlBlur.pack(fill=tk.BOTH, expand=True)
pnlBlur.pack_forget()

# TRACK BAR BLUR KX
tbBlurX = tk.Scale(pnlBlur, from_=0, to=100, resolution=1, bg='#303030', fg='White', highlightthickness=0,
                   orient='horizontal', command=tbBlurX_ValueChangedEvent)
tbBlurX.set(0)
tbBlurX.pack(fill=tk.X, padx=20)

# TRACK BAR BLUR KY
tbBlurY = tk.Scale(pnlBlur, from_=0, to=100, resolution=1, bg='#303030', fg='White', highlightthickness=0,
                   orient='horizontal', command=tbBlurY_ValueChangedEvent)
tbBlurY.set(0)
tbBlurY.pack(fill=tk.X, padx=20)

# PANEL RADIOBUTTONS BLUR
pnlBlurRB = tk.Frame(pnlBlur, bg='#303030')
pnlBlurRB.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

# RADIOBUTTON GAUSSIAN BLUR
rbBlurGaussian = tk.Radiobutton(pnlBlurRB, text='Gaussian Blur', variable=rbBlurValue, value='gaussian', bg='#353535',
                                fg='White', activebackground='#353535', activeforeground='White', selectcolor='#353535', command=rbBlur_ClickEvent)
rbBlurGaussian.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# RADIOBUTTON AVERAGING BLUR
rbBlurAveraging = tk.Radiobutton(
    pnlBlurRB, text='Averaging Blur', variable=rbBlurValue, value='averaging', bg='#353535', fg='White', activebackground='#353535', activeforeground='White', selectcolor='#353535', command=rbBlur_ClickEvent)
rbBlurAveraging.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# RADIOBUTTON MEDIAN BLUR
rbBlurMedian = tk.Radiobutton(
    pnlBlurRB, text='Median Blue', variable=rbBlurValue, value='median', bg='#353535', fg='White', activebackground='#353535', activeforeground='White', selectcolor='#353535', command=rbBlur_ClickEvent)
rbBlurMedian.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTTON SAVE
btnBlurSave = tk.Button(pnlBlur, text='Save', bg='#353535', fg='White',
                        command=btnBlurSave_ClickEvent)
btnBlurSave.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(5, 20))

# BUTTON RESET
btnBlurReset = tk.Button(pnlBlur, text='Reset', bg='#353535', fg='White',
                         command=btnBlurReset_ClickEvent)
btnBlurReset.pack(fill=tk.X, side=tk.BOTTOM, padx=20)

# endregion

# region BRIGHTNESS
# PANEL BRIGHTNESS
pnlBns = tk.Frame(opPanel, bg='#303030')
pnlBns.pack(fill=tk.BOTH, expand=True)
pnlBns.pack_forget()

# TRACK BAR BRIGHTNESS
tbBns = tk.Scale(pnlBns, from_=-255, to=255, resolution=1, bg='#303030', fg='White', highlightthickness=0,
                 orient='horizontal', command=tbBns_ValueChangedEvent)
tbBns.set(0)
tbBns.pack(fill=tk.X, padx=20)
# PANEL ALPHA VALUE
pnlBnsValue = tk.Frame(pnlBns, bg='#303030')
pnlBnsValue.pack(fill=tk.X)

# LABEL ALPHA
lblBnsValue = tk.Label(pnlBnsValue, bg='#353535',
                       fg='White', text='Beta Value : 1.0')
lblBnsValue.pack(side=tk.LEFT, padx=20, pady=(10, 5))

# BUTTON SAVE
btnBnsSave = tk.Button(
    pnlBns, text='Save', bg='#353535', fg='White', command=btnBnsSave_ClickEvent)
btnBnsSave.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(5, 20))

# BUTTON RESET
btnBnsReset = tk.Button(
    pnlBns, text='Reset', bg='#353535', fg='White', command=btnBnsReset_ClickEvent)
btnBnsReset.pack(fill=tk.X, side=tk.BOTTOM, padx=20)

# endregion

# region CROP
# PANEL CROP
pnlCrop = tk.Frame(opPanel, bg='#303030')
pnlCrop.pack(fill=tk.BOTH, expand=True)
pnlCrop.pack_forget()

# TRACK BAR CROP
tbCrop = tk.Scale(pnlCrop, from_=0, to=99, resolution=1, bg='#303030', fg='White', highlightthickness=0,
                  orient='horizontal', command=tbCrop_ValueChangedEvent)
tbCrop.set(0)
tbCrop.pack(fill=tk.X, padx=20)

# PANEL RADIOBUTTONS CROP
pnlCropRB = tk.Frame(pnlCrop, bg='#303030')
pnlCropRB.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

# RADIOBUTTON LEFT CROP
rbCropLeft = tk.Radiobutton(
    pnlCropRB, text='Left', variable=rbCropValue, value='left', bg='#353535', fg='White', activebackground='#353535', activeforeground='White', selectcolor='#353535', command=rbCrop_ClickEvent)
rbCropLeft.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# RADIOBUTTON RIGHT CROP
rbCropRight = tk.Radiobutton(
    pnlCropRB, text='Right', variable=rbCropValue, value='right', bg='#353535', fg='White', activebackground='#353535', activeforeground='White', selectcolor='#353535', command=rbCrop_ClickEvent)
rbCropRight.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# RADIOBUTTON TOP CROP
rbCropTop = tk.Radiobutton(
    pnlCropRB, text='Top', variable=rbCropValue, value='top', bg='#353535', fg='White', activebackground='#353535', activeforeground='White', selectcolor='#353535', command=rbCrop_ClickEvent)
rbCropTop.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# RADIOBUTTON BOTTOM CROP
rbCropBottom = tk.Radiobutton(
    pnlCropRB, text='Bottom', variable=rbCropValue, value='bottom', bg='#353535', fg='White', activebackground='#353535', activeforeground='White', selectcolor='#353535', command=rbCrop_ClickEvent)
rbCropBottom.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# RADIOBUTTON HORIZONTAL CROP
rbCropHori = tk.Radiobutton(pnlCropRB, text='Horizontal', bg='#353535', fg='White', activebackground='#353535', activeforeground='White', selectcolor='#353535',
                            variable=rbCropValue, value='horizontal', command=rbCrop_ClickEvent)
rbCropHori.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# RADIOBUTTON VERTICAL CROP
rbCropVert = tk.Radiobutton(pnlCropRB, text='Vertical', bg='#353535', fg='White', activebackground='#353535', activeforeground='White', selectcolor='#353535',
                            variable=rbCropValue, value='vertical', command=rbCrop_ClickEvent)
rbCropVert.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# RADIOBUTTON ALL CROP
rbCropAll = tk.Radiobutton(
    pnlCropRB, text='All', variable=rbCropValue, value='all', bg='#353535', fg='White', activebackground='#353535', activeforeground='White', selectcolor='#353535', command=rbCrop_ClickEvent)
rbCropAll.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTTON SAVE
btnCropSave = tk.Button(pnlCrop, text='Save', bg='#353535', fg='White',
                        command=btnCropSave_ClickEvent)
btnCropSave.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(5, 20))

# BUTTON RESET
btnCropReset = tk.Button(pnlCrop, text='Reset', bg='#353535', fg='White',
                         command=btnCropReset_ClickEvent)
btnCropReset.pack(fill=tk.X, side=tk.BOTTOM, padx=20)

# endregion

# region INVERT
# PANEL INVERT
pnlInvert = tk.Frame(opPanel, bg='#303030')
pnlInvert.pack(fill=tk.BOTH, expand=True)
pnlInvert.pack_forget()


# BUTTON RESET
btnInvertReset = tk.Button(pnlInvert, text='Reset', bg='#353535', fg='White',
                           command=btnInvertReset_ClickEvent)
btnInvertReset.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(5, 20))

# BUTTON SAVE
btnInvertSave = tk.Button(pnlInvert, text='Save', bg='#353535', fg='White',
                          command=btnInvertSave_ClickEvent)
btnInvertSave.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(5, 0))

# BUTTON doINVERT
btnDoInvert = tk.Button(pnlInvert, text='Invert', bg='#353535', fg='White',
                        command=btnDoInvert_ClickEvent)
btnDoInvert.pack(fill=tk.X, expand=True, side=tk.BOTTOM, padx=20, pady=(20, 0))


# endregion

# region MIRROR
# PANEL MIRROR
pnlMirror = tk.Frame(opPanel, bg='#303030')
pnlMirror.pack(fill=tk.BOTH, expand=True)
pnlMirror.pack_forget()

# PANEL MIRROR BUTTONS
pnlMirrorBtns = tk.Frame(pnlMirror, bg='#303030')
pnlMirrorBtns.pack(fill=tk.BOTH, expand=True, padx=20, pady=(15, 5))

# BUTON X AXIS
btnMirrorX = tk.Button(pnlMirrorBtns, text='X Axis', bg='#353535', fg='White',
                       command=lambda: op.Mirror('x'))
btnMirrorX.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Y AXIS
btnMirrorY = tk.Button(pnlMirrorBtns, text='Y Axis', bg='#353535', fg='White',
                       command=lambda: op.Mirror('y'))
btnMirrorY.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON XY AXIS
btnMirrorXY = tk.Button(pnlMirrorBtns, text='X&Y Axis', bg='#353535', fg='White',
                        command=lambda: op.Mirror('xy'))
btnMirrorXY.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTTON SAVE
btnMirrorSave = tk.Button(pnlMirror, text='Save', bg='#353535', fg='White',
                          command=btnMirrorSave_ClickEvent)
btnMirrorSave.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(5, 20))

# endregion

# region ROTATE
# PANEL ROTATE
pnlRotate = tk.Frame(opPanel, bg='#303030')
pnlRotate.pack(fill=tk.BOTH, expand=True)
pnlRotate.pack_forget()

# TRACK BAR ROTATE
tbRotate = tk.Scale(pnlRotate, from_=0, to=360, bg='#303030', fg='White', highlightthickness=0,
                    resolution=10, orient='horizontal', command=tbRotate_ValueChangedEvent)
tbRotate.set(0)
tbRotate.pack(fill=tk.X, padx=20)

# PANEL DEGREE VALUE
pnlDegree = tk.Frame(pnlRotate, bg='#303030')
pnlDegree.pack(fill=tk.X)

# LABEL DEGREE
lblDegree = tk.Label(pnlDegree, bg='#303030', fg='White', text='Degree : 0')
lblDegree.pack(side=tk.LEFT, padx=20, pady=(10, 5))

# BUTTON SAVE
btnRotateSave = tk.Button(pnlRotate, text='Save', bg='#353535', fg='White',
                          command=btnRotateSave_ClickEvent)
btnRotateSave.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(5, 20))

# BUTTON RESET
btnRotateReset = tk.Button(pnlRotate, text='Reset', bg='#353535', fg='White',
                           command=btnRotateReset_ClickEvent)
btnRotateReset.pack(fill=tk.X, side=tk.BOTTOM, padx=20)

# endregion

# region AUTOMATIC IMAGE ENHANCEMENT
# PANEL AUTOMATIC IMAGE ENHANCEMENT
pnlAIE = tk.Frame(opPanel, bg='#303030')
pnlAIE.pack(fill=tk.BOTH, expand=True)
pnlAIE.pack_forget()

# BUTTON AUTO BRIGHTNESS AND CONTRAST
btnAutoBrightAndContrast = tk.Button(pnlAIE, text = 'Brightness and Contrast Regulator',bg='#353535', fg='White', command=btnAutoBrightAndContrast_ClickEvent)
btnAutoBrightAndContrast.pack(fill = tk.X, expand = True, padx=20,pady=5)

# BUTTON SAVE
btnAIESave = tk.Button(
    pnlAIE, text='Save', bg='#353535', fg='White', command=btnAIESave_ClickEvent)
btnAIESave.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(5, 20))

# BUTTON RESET
btnAIEReset = tk.Button(
    pnlAIE, text='Reset', bg='#353535', fg='White', command=btnAIEReset_ClickEvent)
btnAIEReset.pack(fill=tk.X, side=tk.BOTTOM, padx=20)
# endregion

# region CONTRAST

# PANEL CONTRAST
pnlContrast = tk.Frame(opPanel, bg='#303030')
pnlContrast.pack(fill=tk.BOTH, expand=True)
pnlContrast.pack_forget()

# TRACK BAR CONTRAST
tbContrast = tk.Scale(pnlContrast, from_=0.0, to=3.0, bg='#303030', fg='White', highlightthickness=0,
                      resolution=0.1, orient='horizontal', command=tbContrast_ValueChangedEvent)
tbContrast.set(1.0)
tbContrast.pack(fill=tk.X, padx=20)

# PANEL ALPHA VALUE
pnlAlpha = tk.Frame(pnlContrast, bg='#303030')
pnlAlpha.pack(fill=tk.X)

# LABEL ALPHA
lblAlpha = tk.Label(pnlAlpha, bg='#353535', fg='White',
                    text='Alpha Value : 1.0')
lblAlpha.pack(side=tk.LEFT, padx=20, pady=(10, 5))

# BUTTON SAVE
btnContrastSave = tk.Button(
    pnlContrast, text='Save', bg='#353535', fg='White', command=btnContrastSave_ClickEvent)
btnContrastSave.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(5, 20))

# BUTTON RESET
btnContrastReset = tk.Button(
    pnlContrast, text='Reset', bg='#353535', fg='White', command=btnContrastReset_ClickEvent)
btnContrastReset.pack(fill=tk.X, side=tk.BOTTOM, padx=20)
# endregion

# region ARTISTIC FILTERS
# Panel FILTERS
pnlFilters = tk.Frame(opPanel, bg='#303030')
pnlFilters.pack(fill=tk.BOTH, expand=True)
pnlFilters.pack_forget()

# PANEL FILTER BUTTONS1
pnlFilterBtns1 = tk.Frame(pnlFilters, bg='#303030')
pnlFilterBtns1.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

# PANEL FILTER BUTTONS2
pnlFilterBtns2 = tk.Frame(pnlFilters, bg='#303030')
pnlFilterBtns2.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

# PANEL FILTER BUTTONS3
pnlFilterBtns3 = tk.Frame(pnlFilters, bg='#303030')
pnlFilterBtns3.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

# PANEL FILTER BUTTONS4
pnlFilterBtns4 = tk.Frame(pnlFilters, bg='#303030')
pnlFilterBtns4.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

# BUTON Filter 1
btnFilter1 = tk.Button(pnlFilterBtns1, text='Downside Up',
                       bg='#353535', fg='White', command=btnDownsideUpFilter_ClickEvent)
btnFilter1.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 2
btnFilter2 = tk.Button(pnlFilterBtns1, text='Soft B&W',
                       bg='#353535', fg='White', command=btnSoftBWFilter_ClickEvent)
btnFilter2.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 3
btnFilter3 = tk.Button(pnlFilterBtns1, text='Cartoonizer',
                       bg='#353535', fg='White', command=btnCartoonizerEffectFilter_ClickEvent)
btnFilter3.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 4
btnFilter4 = tk.Button(pnlFilterBtns1, text='Ashe',
                       bg='#353535', fg='White', command=btnAsheFilter_ClickEvent)
btnFilter4.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 5
btnFilter5 = tk.Button(pnlFilterBtns1, text='BRoss',
                       bg='#353535', fg='White', command=btnBRossFilter_ClickEvent)
btnFilter5.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 6
btnFilter6 = tk.Button(pnlFilterBtns1, text='Cool',
                       bg='#353535', fg='White', command=btnCoolFilter_ClickEvent)
btnFilter6.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 7
btnFilter7 = tk.Button(pnlFilterBtns2, text='Negative',
                       bg='#353535', fg='White', command=btnNegativeFilter_ClickEvent)
btnFilter7.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 8
btnFilter8 = tk.Button(pnlFilterBtns2, text='Carbon Paper',
                       bg='#353535', fg='White', command=btnCarbonPaperFilter_ClickEvent)
btnFilter8.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 9
btnFilter9 = tk.Button(pnlFilterBtns2, text='Warm',
                       bg='#353535', fg='White', command=btnWarmFilter_ClickEvent)
btnFilter9.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 10
btnFilter10 = tk.Button(pnlFilterBtns2, text='Master Sketch',
                        bg='#353535', fg='White', command=btnMasterSketch_ClickEvent)
btnFilter10.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 11
btnFilter11 = tk.Button(pnlFilterBtns2, text='Colored Master Sketch',
                        bg='#353535', fg='White', command=btnColoredMasterSketch_ClickEvent)
btnFilter11.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 12
btnFilter12 = tk.Button(pnlFilterBtns2, text='Emboss',
                        bg='#353535', fg='White', command=btnEmboss_ClickEvent)
btnFilter12.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 13
btnFilter13 = tk.Button(pnlFilterBtns3, text='Downside Neon',
                        bg='#353535', fg='White', command=btnDownsideNeon_ClickEvent)
btnFilter13.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 14
btnFilter14 = tk.Button(pnlFilterBtns3, text='Marked',
                        bg='#353535', fg='White', command=btnMarked_ClickEvent)
btnFilter14.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 15
btnFilter15 = tk.Button(pnlFilterBtns3, text='Black Sunny',
                        bg='#353535', fg='White', command=btnblackSunny_ClickEvent)
btnFilter15.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 16
btnFilter16 = tk.Button(pnlFilterBtns3, text='Vivid Neon',
                        bg='#353535', fg='White', command=btnvividNeon_ClickEvent)
btnFilter16.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 17
btnFilter17 = tk.Button(pnlFilterBtns3, text='Lala',
                        bg='#353535', fg='White', command=btnlala_ClickEvent)
btnFilter17.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 18
btnFilter18 = tk.Button(pnlFilterBtns3, text='Wonderland',
                        bg='#353535', fg='White', command=btnwonderland_ClickEvent)
btnFilter18.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 19
btnFilter19 = tk.Button(pnlFilterBtns4, text='Sundown',
                        bg='#353535', fg='White', command=btnsundown_ClickEvent)
btnFilter19.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 20
btnFilter20 = tk.Button(pnlFilterBtns4, text='Glossy',
                        bg='#353535', fg='White', command=btnglossy_ClickEvent)
btnFilter20.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 21
btnFilter21 = tk.Button(pnlFilterBtns4, text='Hand-Drawn',
                        bg='#353535', fg='White', command=btnhandDrawn_ClickEvent)
btnFilter21.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 22
btnFilter22 = tk.Button(pnlFilterBtns4, text='Old Town',
                        bg='#353535', fg='White', command=btnoldTown_ClickEvent)
btnFilter22.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 23
btnFilter23 = tk.Button(pnlFilterBtns4, text='At-Dawn',
                        bg='#353535', fg='White', command=btnatDawn_ClickEvent)
btnFilter23.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# BUTON Filter 24
btnFilter24 = tk.Button(pnlFilterBtns4, text='Candy Girl',
                        bg='#353535', fg='White', command=btncandyGirl_ClickEvent)
btnFilter24.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)


# BUTTON SAVE
btnFilterSave = tk.Button(pnlFilters, text='Save', bg='#353535', fg='White',
                          command=btnFilterSave_ClickEvent)
btnFilterSave.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(5, 20))

# BUTTON RESET
btnFilterReset = tk.Button(pnlFilters, text='Reset', bg='#353535', fg='White',
                           command=btnFilterReset_ClickEvent)
btnFilterReset.pack(fill=tk.X, side=tk.BOTTOM, padx=20)

# endregion

# region START MAIN LOOP
master.config(bg='#454545', menu=menubar)
master.iconbitmap('favicon.ico')
master.mainloop()
# endregion
