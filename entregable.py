import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from fpdf import FPDF  # Asegúrate de instalar con: pip install fpdf

class RFAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RF Signal Analyzer")
        self.files = []

        # Crear el marco principal
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Crear el panel izquierdo de color negro
        left_panel = tk.Frame(main_frame, bg="black", width=200)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)

        # Crear botones en el panel izquierdo
        self.open_button = tk.Button(left_panel, text="Abrir Archivos CSV", command=self.open_files, bg="gray", fg="white")
        self.open_button.pack(pady=10, padx=10, fill=tk.X)

        self.analyze_button = tk.Button(left_panel, text="Analizar Señal", command=self.analyze_signals, bg="gray", fg="white")
        self.analyze_button.pack(pady=10, padx=10, fill=tk.X)

        self.export_button = tk.Button(left_panel, text="Descargar Informe", command=self.export_report, bg="gray", fg="white")
        self.export_button.pack(pady=10, padx=10, fill=tk.X)

        # Crear el panel derecho para mostrar los resultados
        self.results_frame = tk.Frame(main_frame)
        self.results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.results_text = tk.Text(self.results_frame, height=20, width=80)
        self.results_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

    def open_files(self):
        self.files = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if self.files:
            messagebox.showinfo("Archivos Cargados", f"Se han cargado {len(self.files)} archivos.")

    def analyze_signals(self):
        if not self.files:
            messagebox.showwarning("Advertencia", "No se han cargado archivos.")
            return

        results = []
        for file in self.files:
            try:
                df = pd.read_csv(file)
                frequencies = df.iloc[:, 0].values
                timestamps = df.columns[1:]

                for timestamp in timestamps:
                    amplitudes = df[timestamp].values
                    smoothed_amplitudes = savgol_filter(amplitudes, window_length=11, polyorder=2)

                    # Calcular parámetros
                    central_freq_index = np.argmax(smoothed_amplitudes)
                    central_freq = frequencies[central_freq_index]
                    half_max = np.max(smoothed_amplitudes) - 3
                    bandwidth = self.calculate_bandwidth(frequencies, smoothed_amplitudes, half_max)
                    peak_amplitude = np.max(smoothed_amplitudes)
                    noise_level = np.mean(smoothed_amplitudes[smoothed_amplitudes < np.percentile(smoothed_amplitudes, 20)])
                    snr = self.calculate_snr(smoothed_amplitudes, noise_level)
                    crest_factor = peak_amplitude / np.sqrt(np.mean(np.square(smoothed_amplitudes)))

                    # Guardar resultados
                    results.append(f"Archivo: {file}, Timestamp: {timestamp}")
                    results.append(f"Frecuencia Central: {central_freq:.2f} Hz")
                    results.append(f"Ancho de Banda (BW): {bandwidth:.2f} Hz")
                    results.append(f"Amplitud/Potencia: {peak_amplitude:.2f} dB")
                    results.append(f"Nivel de Ruido: {noise_level:.2f} dB")
                    results.append(f"SNR: {snr:.2f} dB")
                    results.append(f"Crest Factor: {crest_factor:.2f}")
                    results.append("")

            except Exception as e:
                messagebox.showerror("Error", f"Error al procesar {file}: {str(e)}")
                continue

        if results:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "\n".join(results))
            self.results = results  # Guardar resultados para exportar
        else:
            messagebox.showinfo("Sin Resultados", "No se encontraron resultados para mostrar.")

    def calculate_snr(self, signal, noise_level):
        signal_power = np.max(signal)
        snr = 10 * np.log10(signal_power / noise_level)
        return snr

    def calculate_bandwidth(self, frequencies, amplitudes, half_max):
        indices = np.where(amplitudes > half_max)[0]
        if len(indices) > 0:
            bandwidth = frequencies[indices[-1]] - frequencies[indices[0]]
        else:
            bandwidth = 0
        return bandwidth

    def export_report(self):
        if hasattr(self, 'results') and self.results:
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="Informe de Análisis de Señal RF", ln=True, align='C')

                for line in self.results:
                    pdf.cell(0, 10, txt=line, ln=True)

                save_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
                if save_path:
                    pdf.output(save_path)
                    messagebox.showinfo("Informe Guardado", "El informe se ha guardado correctamente.")
            except Exception as e:
                messagebox.showerror("Error", f"Error al guardar el informe: {str(e)}")
        else:
            messagebox.showwarning("Advertencia", "No hay resultados para exportar.")

if __name__ == "__main__":
    root = tk.Tk()
    app = RFAnalyzerApp(root)
    root.mainloop()

