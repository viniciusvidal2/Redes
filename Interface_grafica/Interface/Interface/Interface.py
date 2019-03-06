
import tkinter as tk


LARGE_FONT = ("Verdana",12)

class SeaofBTCapp(tk.Tk):
    def __init__(self, *args,**kwargs):

        tk.Tk.__init__(self, *args,**kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand=True)

        container.rowconfigure(0, weight=1)
        container.grid_columnconfigure(0,weight=1)

        self.frames = {}

        for F in(StartPage, PageOne, PageTwo):

            frame = F(container,self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


def qf(param):
    print(param)

class StartPage(tk.Frame):

   def __init__(self, parent, controller):

       tk.Frame.__init__(self,parent)
       label = tk.Label(self, text = "Start Page",font = LARGE_FONT )
       label.pack(pady=10,padx=10)

       button1 = tk.Button(self, text="Visite Page 1", command= lambda: controller.show_frame(PageOne))
       button1.pack()

       button1 = tk.Button(self, text="Visite Page 2", command= lambda: controller.show_frame(PageTwo))
       button1.pack()


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text = "Page One",font = LARGE_FONT )
        label.pack(pady=10,padx=10)
        button1 = tk.Button(self, text="Back to home", command= lambda: controller.show_frame(StartPage))
        button1.pack()


class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text = "Page Two",font = LARGE_FONT )
        label.pack(pady=10,padx=10)
        button1 = tk.Button(self, text="Back to home", command= lambda: controller.show_frame(StartPage))
        button1.pack()





app = SeaofBTCapp()
app.mainloop()






#class BuckysButtons:
#    def __init__(self, master):
        
#        #frame = Frame(master)
#        #frame.pack()
        
#        self.printButton = Button(master,text="Print message", command=self.printMessage)
#        self.printButton.pack(side=TOP)

#        self.quitButton = Button(master, text="Quit", command=master.quit)
#        self.quitButton.pack(side=TOP)

#    def printMessage(self):
#        print("It is working")

#class MyFrame:
#    def __init__(self,master):

#        self.Tops = Frame(master,width = 1350,height = 100, bd = 1, relief = "raise")
#        self.Tops.pack(side=TOP)
#        self.Tops.configure(background='gray')

#        self.Right = Frame(master,width = 200,height = 600, bd = 2, relief = "raise")
#        self.Right.pack(side=RIGHT)

#        self.Left = Frame(master,width = 1145,height = 600, bd = 2, relief = "raise")
#        self.Left.pack(side=LEFT)



#root = Tk()
#root.geometry("1350x750+0+0")
#root.title("Página Inicial")
#root.configure(background ='blue')

## ========================= FRAME TOP ===================================

#Tops = Frame(root,width = 1350,height = 100, bd = 1, relief = "raise")
#Tops.pack(side=TOP)
#Tops.configure(background='gray')

#FrameRight = Frame(root,width = 200,height = 640, bd = 2, relief = "raise")
#FrameRight.pack(side=RIGHT)

#FrameLeft = Frame(root,width = 1145,height = 640, bd = 2, relief = "raise")
#FrameLeft.pack(side=LEFT)

## ============================ Labels =====================================

#lbl  = Label(Tops,font=('arial',35,'bold'),text = 'Bolsa de Valores', bd=8, width=50)
#lbl.pack(side = TOP)    

## ============================  VARIAVEIS  ===================================
#chkval = IntVar()
#chkval.set("0")
#chk1 = Checkbutton(FrameRight ,text = "Plot", variable = chkval, onvalue=1, offvalue=0,font=('arial',10,'bold')).pack(side=TOP)




#root.mainloop ()