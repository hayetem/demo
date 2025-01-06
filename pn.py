import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array




def load_and_predict_image(model, label_frame):
    file_path = filedialog.askopenfilename(
        title="Sélectionner une image",
        filetypes=[("Image files", ".jpg;.jpeg;*.png")]
    )
    if file_path:
        try:
            # Charger et prétraiter l'image
            img = load_img(file_path, target_size=(227, 227))  # Redimensionner l'image
            img_array = img_to_array(img) / 255.0  # Normaliser l'image
            img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch

            # Faire une prédiction
            prediction = model.predict(img_array)
            predicted_class = (prediction > 0.5).astype(int)[0][0]  # Binariser la prédiction
            class_labels = {1: "no_belt", 0: "belt"}
            predicted_label = class_labels[predicted_class]

            # Supprimer les widgets précédents du frame (si existants)
            for widget in label_frame.winfo_children():
                widget.destroy()

            # Cadre pour l'image
            img_frame = tk.Frame(label_frame, bg="black", bd=2)  # Cadre noir avec bordure fine
            img_frame.pack(pady=10)

            # Afficher l'image
            img_display = Image.open(file_path)
            img_display = img_display.resize((200, 200))  # Taille pour l'affichage
            img_tk = ImageTk.PhotoImage(img_display)
            img_label = tk.Label(img_frame, image=img_tk, bg="white")  # Fond blanc pour l'image
            img_label.image = img_tk
            img_label.pack()

            # Ajouter le texte de prédiction
            prediction_label = tk.Label(
                label_frame,
                text=f"Classe prédite : {predicted_label}",
                font=("Arial", 14, "bold"),
                fg="green" if predicted_label == "belt" else "red",
                bg="white"
            )
            prediction_label.pack(pady=5)

        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur est survenue lors de la prédiction : {str(e)}")

def create_icon_button(root, image_path, command=None):
    try:
    
        image = Image.open(image_path)
        image = image.resize((50, 50))  # Redimensionner l'image
        icon = ImageTk.PhotoImage(image)

        button = tk.Button(
            root,
            image=icon,  # Ensure the icon is defined earlier in your code
            command=command,  # Ensure the 'command' function is defined
            bg="#F4C7B8",  # Button background color
            activebackground="#F4C7B8",  # Background color when active
            borderwidth=1,  # Slight border for definition
            relief="solid",  # Solid border for a defined look
            font=("Helvetica", 12, "bold"),  # Better font choice
            fg="#333",  # Text color (if any)
            padx=10,  # Horizontal padding
            pady=10,  # Vertical padding
            bd=5,  # Border padding for a 3D effect
            highlightthickness=0,  # Remove highlight border when focused
            width=150,  # Width of the button (adjust as needed)
            height=80,  # Height of the button (adjust as needed)
            overrelief="sunken",  # Effect when hovered
            )

        # Adding a hover effect using event binding
        def on_enter(event):
            button.config(bg="#F1A7B1")  # Lighter shade on hover
        def on_leave(event):
            button.config(bg="#F4C7B8")  # Reset to original color

            button.bind("<Enter>", on_enter)
            button.bind("<Leave>", on_leave)

            button.pack(pady=20)
        button.image = icon  
        return button
    except FileNotFoundError:
        print(f"Erreur : L'image {image_path} est introuvable.")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement de l'image {image_path} : {e}")
        return None
def create_text_button(root, text, command=None, bg="#F4C7B8"):
    def on_enter(event):
        button.config(bg="#F1A7B1")  # Change to a lighter shade on hover

    def on_leave(event):
        button.config(bg=bg)  # Reset to the original color

    button = tk.Button(
        root,
        text=text,
        command=command,
        bg=bg,
        activebackground=bg,
        font=("Arial", 12, "bold"),  # Bold font for better visibility
        fg="#FFFFFF",  # White text color for better contrast
        relief="flat",  # Modern flat relief
        borderwidth=0,  # No visible border
        padx=15,  # Horizontal padding for a spacious look
        pady=8  # Vertical padding for a spacious look
    )

    # Add hover effects
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)

    return button


def load_model_and_data(model_path, test_generator):
    # Charger le modèle
    model = load_model(model_path)
    
    # Faire des prédictions
    y_true = test_generator.classes  
    y_pred_probs = model.predict(test_generator)  
    y_pred = (y_pred_probs > 0.5).astype(int)  
    
    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    return cm, y_true, y_pred


def show_confusion_matrix(root, cm):
    # Création de la figure
    fig, ax = plt.subplots(figsize=(3, 3))  # Taille ajustée pour une meilleure visibilité

    # Création de la matrice de confusion avec un style amélioré
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='coolwarm',  # Palette de couleurs attrayante
        cbar=True,        # Afficher la barre de couleur
        annot_kws={"size": 14},  # Taille des annotations
        linewidths=1,  # Lignes entre les cellules
        linecolor='gray',  # Couleur des lignes
        ax=ax
    )

    # Personnalisation des axes
    ax.set_title('Matrice de Confusion', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Prédictions', fontsize=14, labelpad=10)
    ax.set_ylabel('Vérités', fontsize=14, labelpad=10)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.xaxis.set_ticklabels(['Classe 0', 'Classe 1'])  # Exemple d'étiquettes (modifiez selon vos données)
    ax.yaxis.set_ticklabels(['Classe 0', 'Classe 1'])

    # Suppression des bordures blanches inutiles
    plt.tight_layout()

    # Intégration dans tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=0, columnspan=4, pady=20, padx=20)  # Centrage avec des marges
    canvas.draw()


def display_image_croi_from_path():
   
    img_path = r'C:\Users\hayet\Music\ai\croi.jpg'  
    try:
        img = Image.open(img_path)
        img = img.resize((200, 200))  
        img_tk = ImageTk.PhotoImage(img)
        image_label2.config(image=img_tk)
        image_label2.image = img_tk  
    except FileNotFoundError:
        print(f"Erreur : L'image au chemin {img_path} est introuvable.")
    except Exception as e:
        print(f"Erreur lors du chargement de l'image : {e}")

def display_image_decroi_from_path():
   
    im_path = r'C:\Users\hayet\Music\ai\decroi.jpg'  
    try:
        img = Image.open(im_path)
        img = img.resize((200, 200))  
        img_tk = ImageTk.PhotoImage(img)

        # Mettre à jour l'image dans le widget label
        image_label3.config(image=img_tk)
        image_label3.image = img_tk  
    except FileNotFoundError:
        print(f"Erreur : L'image au chemin {im_path} est introuvable.")
    except Exception as e:
        print(f"Erreur lors du chargement de l'image : {e}")


def show_f1_score(root, y_true, y_pred):
    f1 = f1_score(y_true, y_pred) 
    f1_label.config(text=f"F1 Score : {f1:.2f}")  

def show_recall(root, y_true, y_pred):
    recall = recall_score(y_true, y_pred)  
    recall_label.config(text=f"Recall : {recall:.2f}")  

def show_accuracy(root, y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)  
    accuracy_label.config(text=f"Accuracy : {accuracy:.2f}") 

def main():
   
    model_path = r'C:\Users\hayet\Music\ai\saved_model.h5'
    test_dir = r'C:\Users\hayet\Music\ai\test\test'
    model = load_model(model_path)
    # Préparation des générateurs de données
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(227, 227),
        batch_size=32,
        class_mode='binary'
    )

    # Charger le modèle et calculer la matrice de confusion
    cm, y_true, y_pred = load_model_and_data(model_path, test_generator)

    # Créer la fenêtre principale
    root = tk.Tk()
    root.title("projet CNN Dalel&Hayetem")

    
    try:
        default_image = Image.open(r'C:\Users\hayet\Music\ai\init.jpg')  
        default_image = default_image.resize((200, 200))
        default_image_tk = ImageTk.PhotoImage(default_image)
        default_image1 = Image.open(r'C:\Users\hayet\Music\ai\icon3.png')  
        default_image1 = default_image1.resize((200, 200))
        default_image_tk1 = ImageTk.PhotoImage(default_image1)
        default_image5 = Image.open(r'C:\Users\hayet\Music\ai\icon4.png')  
        default_image5 = default_image5.resize((200, 200))
        default_image_tk5 = ImageTk.PhotoImage(default_image5)
    except FileNotFoundError:
        print("Erreur : L'image par defaut n'a pas ete etrouvee.")
        default_image_tk = None

    global image_label
    image_label = tk.Label(root, image=default_image_tk)
    image_label.image = default_image_tk  # Garde une référence pour éviter le garbage collector
    image_label.grid(row=0, column=4, columnspan=4, pady=20)

    global image_label2, image_label3
    
    image_label2 = tk.Label(root, image=default_image_tk1)
    image_label2.image = default_image_tk1
    image_label2.grid(row=1, column=0, columnspan=3, pady=20)


    image_label3 = tk.Label(root, image=default_image_tk5)
    image_label3.image = default_image_tk5
    image_label3.grid(row=1, column=4, columnspan=4, pady=20)

    cm_label = tk.Label(root, text="Matrice de Confusion")
    cm_label.grid(row=0, column=0, columnspan=3, pady=20)

    icon1_path = r'C:\Users\hayet\Music\ai\icon1.png'
    icon2_path = r'C:\Users\hayet\Music\ai\icon2.png'
    icon3_path = r'C:\Users\hayet\Music\ai\icon3.png'
    icon4_path = r'C:\Users\hayet\Music\ai\icon4.png'

    # Boutons

    button1 = create_icon_button(root, icon1_path, command=lambda: show_confusion_matrix(root, cm))
    if button1:
        button1.grid(row=4, column=0, padx=10, pady=10)

    button2 = create_icon_button(root, icon2_path, command=lambda: load_and_predict_image(load_model(model_path), image_label)) 
    if button2:
        button2.grid(row=4, column=2, padx=10, pady=10)

    button3 = create_icon_button(root, icon3_path, command=display_image_croi_from_path) 
    if button3:
        button3.grid(row=4, column=4, padx=10, pady=10)

    button4 = create_icon_button(root, icon4_path, command=display_image_decroi_from_path)  
    if button4:
        button4.grid(row=4, column=6, padx=10, pady=10)

    button5 = create_text_button(root, text="Recall", command=lambda: show_recall(root, y_true, y_pred)) 
    if button5:
        button5.grid(row=3, column=3, padx=10, pady=10)
    button6 = create_text_button(root, text="F1_Score", command=lambda: show_f1_score(root, y_true, y_pred))  
    if button6:
        button6.grid(row=3, column=1, padx=10, pady=10)

    button7 = create_text_button(root, text="Accuracy", command=lambda: show_accuracy(root, y_true, y_pred))
    if button7:
        button7.grid(row=3, column=5, padx=10, pady=10)

    global f1_label, recall_label
    f1_label = tk.Label(root, text="F1 Score : ", font=("Arial", 12))
    f1_label.grid(row=2, column=1, columnspan=2, pady=10)

    recall_label = tk.Label(root, text="Recall : ", font=("Arial", 12))
    recall_label.grid(row=2, column=3, columnspan=2, pady=10)
    global accuracy_label
    accuracy_label = tk.Label(root, text="Accuracy : ", font=("Arial", 12))
    accuracy_label.grid(row=2, column=5, columnspan=2, pady=10)


    # Lancer l'application
    root.mainloop()


if __name__ == "__main__":
    main()