from tkinter import *
from tkinter import ttk
from tkinter import Tk
from WheatYield_xx import WheatYield
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog, messagebox
import sys
from threading import Thread


class MainWindow:
    root = Tk()

    def __init__(self):
        self.label = Label(self.root, text='High Throughput Phenotyping of various crops', font=('Arial', 12))
        self.buttonWheat = Button(self.root, text='Wheat', padx=20, pady=20, font=('Arial', 12),
                                  command=self.wheat_button)
        self.buttonRice = Button(self.root, text='Rice', padx=20, pady=20, font=('Arial', 12), command=self.rice_button)

    @staticmethod
    def rice_button():
        messagebox.showinfo(title='Rice', message='Phenotyping of Rice has not been implemented yet')

    @staticmethod
    def wheat_button():
        WheatWindow()

    def run(self):
        self.root.geometry('400x300')
        self.root.title('Phenotyping')

        self.label.pack(padx=10, pady=10)
        self.buttonWheat.pack(padx=10, pady=10)
        #self.buttonRice.pack(padx=10, pady=20)

        self.root.mainloop()


class WheatWindow:
    def __init__(self):
        self.window = Toplevel()
        self.window.geometry('100x100')
        self.window.title('Wheat')

        self.button = Button(self.window, text='Select wheat field image',
                             font=('Arial', 14),
                             wraplength=100,
                             command=self.button_press)
        self.button.pack(pady=10)

    def button_press(self):
        path, image = self.read_image()
        self.window.destroy()
        ImageWindow(path, image)

    @staticmethod
    def read_image():
        path = filedialog.askopenfilename()
        image = Image.open(path)

        return path, image


class ImageWindow:
    def __init__(self, path, image):
        self.image = image
        self.path = path
        self.width, self.height = self.image.size
        self.resize_image = self.image.resize((int(self.width / 2), int(self.height / 2)))
        self.resize_image = ImageTk.PhotoImage(self.resize_image)

        self.window = Toplevel()
        self.window.geometry(str(int(self.width / 2)) + 'x' + str(int(self.height / 2 + 50)))
        if hasattr(self.image, 'filename'):
            self.window.title(self.image.filename.split('/')[-1])
        else:
            self.window.title('Wheat')

        self.label = Label(self.window, image=self.resize_image)
        self.label.image = self.resize_image
        self.label.pack()
        self.button = Button(self.window, text='Count', font=('Arial', 14), command=self.button_press)
        self.button.pack()
        self.pb = ttk.Progressbar(
            self.window,
            orient='horizontal',
            mode='indeterminate',
            length=int(self.width / 2)
        )

    def start_progress(self):
        self.button.destroy()
        self.pb.pack()
        self.pb.start(20)

    def stop_progress(self):
        self.pb.stop()
        self.window.destroy()

    def button_press(self):
        self.start_progress()

        thread = Detection(self.path, self, MainWindow)
        thread.start()

        self.monitor(thread)

    def monitor(self, thread):
        if thread.is_alive():
            self.window.after(100, lambda: self.monitor(thread))
        else:
            self.stop_progress()
            SpikeWindow(self.image,
                        thread.det_image,
                        thread.density,
                        thread.count,
                        thread.area,
                        thread.est)


class Detection(Thread):
    def __init__(self, path, curr_window, main_window):
        super(Detection, self).__init__()
        self.path = path
        self.curr_window = curr_window
        self.main_window = main_window
        self.estimator = WheatYield(self.path)
        self.det_image = None
        self.density = None
        self.count = None
        self.area = None
        self.est = None

    def run(self):
        self.check_if_alive()
        spikes, spike_results = self.estimator.inference_crop()
        self.det_image = self.estimator.draw_detection(spike_results)
        self.density, self.count, self.area, self.est = self.estimator.estimate_yield(spikes)

    def check_if_alive(self):
        self.main_window.root.after(100, lambda: self.check_if_alive())
        if not self.curr_window.window.winfo_exists():
            print('Should exit')
            self.estimator.run = False


class SpikeWindow:
    def __init__(self,
                 image,
                 det_image,
                 density,
                 count,
                 area,
                 est):
        self.image = image
        self.width, self.height = self.image.size
        self.width = self.width / 2
        self.resize_image = self.image.resize((int(self.width / 2.1), int(self.height / 2)))
        self.resize_image = ImageTk.PhotoImage(self.resize_image)

        if det_image is None:
            if hasattr(self.image, 'filename'):
                messagebox.showinfo(title=self.image.filename.split('/')[-1],
                                    message='No spikes detected!')
            else:
                messagebox.showinfo(title='Wheat', message='No spikes detected!')

        else:
            self.det_image = det_image.resize((int(self.width / 2.1), int(self.height / 2)))
            self.det_image = ImageTk.PhotoImage(self.det_image)

            self.window = Toplevel()
            self.window.geometry(str(int(self.width * 0.96)) + 'x' + str(int(self.height / 2 + 100)))
            if hasattr(self.image, 'filename'):
                self.window.title(self.image.filename.split('/')[-1])
            else:
                self.window.title('Wheat')
            # self.label1 = Label(self.window, image=self.resize_image)
            # self.label1.image = self.resize_image
            # self.label1.grid(row=0, column=0)

            self.label2 = Label(self.window, image=self.det_image)
            self.label2.image = self.det_image
            self.label2.pack()

            self.text = Label(self.window, text=f'Total Spikes: {len(count)}\n'
                                                f'Total Spikelets: {sum(count)}\n'
                                                f'Biggest Spike: {max(area)}\n'
                                                f'Most Dense Spike: {max(density):3.2e}\n')
                                                #f'Estimated yield: {est:.3f}')
            self.text.pack()


def main():
    mainWindow = MainWindow()
    mainWindow.run()
    print('xdD')


if __name__ == '__main__':
    main()
