import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkcalendar import DateEntry
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from fpdf import FPDF

class RFAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RF Signal Analyzer")
        self.files = []
        self.data = pd.DataFrame()

        # Crear el marco principal
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Crear el panel izquierdo de color negro
        left_panel = tk.Frame(main_frame, bg="black", width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)

        # Centrar los elementos en el panel izquierdo
        left_panel_inner = tk.Frame(left_panel, bg="black")
        left_panel_inner.pack(expand=True)

        # Crear botones en el panel izquierdo
        self.open_button = tk.Button(left_panel_inner, text="Abrir Archivos CSV", command=self.open_files, bg="gray", fg="white")
        self.open_button.pack(pady=10, padx=10)

        self.analyze_button = tk.Button(left_panel_inner, text="Analizar Señal", command=self.analyze_signals, bg="gray", fg="white")
        self.analyze_button.pack(pady=10, padx=10)

        self.export_button = tk.Button(left_panel_inner, text="Descargar Informe", command=self.export_report, bg="gray", fg="white")
        self.export_button.pack(pady=10, padx=10)

        # Crear filtros para frecuencia
        self.freq_filter_label = tk.Label(left_panel_inner, text="Seleccione Frecuencia", bg="black", fg="white")
        self.freq_filter_label.pack(pady=5)
        self.freq_combobox = ttk.Combobox(left_panel_inner, state="readonly")
        self.freq_combobox.pack(pady=5, padx=10)

        self.apply_freq_button = tk.Button(left_panel_inner, text="Aplicar Filtro Frecuencia", command=self.apply_freq_filter, bg="gray", fg="white")
        self.apply_freq_button.pack(pady=10, padx=10)

        # Crear filtros para fecha
        self.date_filter_label = tk.Label(left_panel_inner, text="Seleccione Fecha", bg="black", fg="white")
        self.date_filter_label.pack(pady=5)
        self.date_filter_entry = DateEntry(left_panel_inner, date_pattern='yyyy-mm-dd')
        self.date_filter_entry.pack(pady=5, padx=10)

        self.apply_date_button = tk.Button(left_panel_inner, text="Aplicar Filtro Fecha", command=self.apply_date_filter, bg="gray", fg="white")
        self.apply_date_button.pack(pady=10, padx=10)

        # Crear el panel derecho para mostrar los resultados
        self.results_frame = tk.Frame(main_frame)
        self.results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.results_text = tk.Text(self.results_frame, height=20, width=80)
        self.results_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

    def open_files(self):
        self.files = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if self.files:
            messagebox.showinfo("Archivos Cargados", f"Se han cargado {len(self.files)} archivos.")
            self.load_frequencies()

    def load_frequencies(self):
        # Cargar frecuencias de los archivos para el filtro
        try:
            all_frequencies = set()
            for file in self.files:
                df = pd.read_csv(file)
                frequencies = df.iloc[:, 0].values
                all_frequencies.update(frequencies)
            self.freq_combobox['values'] = sorted(list(all_frequencies))
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar frecuencias: {str(e)}")

    def apply_freq_filter(self):
        # Aplicar filtro de frecuencia basado en el valor seleccionado
        selected_freq = self.freq_combobox.get()
        if selected_freq:
            self.selected_freq = float(selected_freq)
            messagebox.showinfo("Filtro Aplicado", f"Filtro de Frecuencia: {self.selected_freq} Hz")
        else:
            self.selected_freq = None

    def apply_date_filter(self):
        # Aplicar filtro de fecha basado en el valor seleccionado
        selected_date = self.date_filter_entry.get_date()
        self.selected_date = selected_date.strftime('%Y-%m-%d')
        messagebox.showinfo("Filtro Aplicado", f"Filtro de Fecha: {self.selected_date}")

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
                    # Aplicar filtro de fecha
                    if hasattr(self, 'selected_date') and self.selected_date not in timestamp:
                        continue

                    amplitudes = df[timestamp].values

                    # Aplicar filtro de frecuencia
                    if hasattr(self, 'selected_freq'):
                        freq_mask = (frequencies == self.selected_freq)
                        filtered_frequencies = frequencies[freq_mask]
                        filtered_amplitudes = amplitudes[freq_mask]
                    else:
                        filtered_frequencies = frequencies
                        filtered_amplitudes = amplitudes

                    # Ajustar dinámicamente window_length para evitar errores
                    if len(filtered_amplitudes) < 3:
                        continue

                    window_length = min(11, len(filtered_amplitudes) // 2 * 2 + 1)
                    polyorder = min(2, window_length - 1)

                    # Suavizado de la señal
                    smoothed_amplitudes = savgol_filter(filtered_amplitudes, window_length=window_length, polyorder=polyorder)

                    # Caracterización de la Señal
                    central_freq_index = np.argmax(smoothed_amplitudes)
                    central_freq = filtered_frequencies[central_freq_index]
                    half_max = np.max(smoothed_amplitudes) - 3
                    bandwidth = self.calculate_bandwidth(filtered_frequencies, smoothed_amplitudes, half_max)
                    peak_amplitude = np.max(smoothed_amplitudes)
                    noise_level = np.mean(smoothed_amplitudes[smoothed_amplitudes < np.percentile(smoothed_amplitudes, 20)])
                    snr = self.calculate_snr(smoothed_amplitudes, noise_level)
                    crest_factor = peak_amplitude / np.sqrt(np.mean(np.square(smoothed_amplitudes)))

                    # Guardar resultados
                    results.append(f"Frecuencia: {central_freq:.2f} Hz, Fecha: {timestamp}")
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
            self.results = results
        else:
            messagebox.showinfo("Sin Resultados", "No se encontraron resultados para mostrar.")

    def calculate_snr(self, signal, noise_level):
        # Cálculo de la relación señal-ruido (SNR)
        signal_power = np.max(signal)
        snr = 10 * np.log10(signal_power / noise_level)
        return snr

    def calculate_bandwidth(self, frequencies, amplitudes, half_max):
        # Cálculo del ancho de banda (BW)
        indices = np.where(amplitudes > half_max)[0]
        if len(indices) > 0:
            bandwidth = frequencies[indices[-1]] - frequencies[indices[0]]
        else:
            bandwidth = 0
        return bandwidth

    def export_report(self):
        # Exportación del informe en PDF
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