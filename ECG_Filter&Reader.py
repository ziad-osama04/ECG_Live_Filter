import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, Scale, HORIZONTAL, StringVar
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from matplotlib.animation import FuncAnimation

class ECGLiveFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Signal Processor with Live Display")
        self.root.geometry("1000x850")
        
        # Create the user interface
        self.create_widgets()
        
        # Signal data
        self.time = None
        self.signal_raw = None
        self.signal_filtered = None
        self.signal_original = None  # Keep a copy of the original signal
        self.sampling_rate = 1000  # Hardcoded sampling rate (Hz)
        
        # Variables for live display
        self.animation = None
        self.is_playing = False
        self.current_index = 0
        self.display_width = 10  # Number of seconds to display on screen
        self.animation_speed = 1.0  # Default display speed
        self.filter_count = 0  # Number of times the filter has been applied
        self.y_scale = 1.0  # Y-axis scale
        
    def create_widgets(self):
        # Button frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        # Load file button
        load_btn = tk.Button(btn_frame, text="Load ECG File", command=self.load_file, width=15, height=2)
        load_btn.grid(row=0, column=0, padx=10)
        
        # Process signal button
        self.process_btn = tk.Button(btn_frame, text="Process Signal", command=self.process_signal, width=15, height=2)
        self.process_btn.grid(row=0, column=1, padx=10)
        
        # Display filter application count
        self.filter_count_var = StringVar()
        self.filter_count_var.set("Filter Applications: 0")
        filter_count_label = tk.Label(btn_frame, textvariable=self.filter_count_var, width=20)
        filter_count_label.grid(row=0, column=2, padx=10)
        
        # Reset filter button
        reset_filter_btn = tk.Button(btn_frame, text="Reset Filter", command=self.reset_filter, width=15, height=2)
        reset_filter_btn.grid(row=0, column=3, padx=10)
        
        # Start/Stop live display button
        self.play_btn = tk.Button(btn_frame, text="Start Live Display", command=self.toggle_live_display, width=15, height=2)
        self.play_btn.grid(row=1, column=0, padx=10, pady=5)
        
        # Save filtered signal button
        save_btn = tk.Button(btn_frame, text="Save Filtered Signal", command=self.save_filtered_signal, width=15, height=2)
        save_btn.grid(row=1, column=1, padx=10, pady=5)
        
        # Settings frame
        settings_frame = tk.Frame(self.root)
        settings_frame.pack(pady=5)
        
        # Display speed settings
        speed_frame = tk.Frame(settings_frame)
        speed_frame.grid(row=0, column=0, padx=20)
        
        tk.Label(speed_frame, text="Display Speed:").pack(side=tk.LEFT)
        self.speed_scale = Scale(speed_frame, from_=0.1, to=5.0, resolution=0.1, 
                                orient=HORIZONTAL, length=150, command=self.update_speed)
        self.speed_scale.set(1.0)
        self.speed_scale.pack(side=tk.LEFT)
        
        # Additional settings frame
        settings_frame2 = tk.Frame(self.root)
        settings_frame2.pack(pady=5)
        
        # Display window size settings
        width_frame = tk.Frame(settings_frame2)
        width_frame.grid(row=0, column=0, padx=20)
        
        tk.Label(width_frame, text="Window Width (Seconds):").pack(side=tk.LEFT)
        self.width_scale = Scale(width_frame, from_=2, to=30, resolution=1, 
                                orient=HORIZONTAL, length=150, command=self.update_display_width)
        self.width_scale.set(10)
        self.width_scale.pack(side=tk.LEFT)
        
        # Y-axis scale settings
        yscale_frame = tk.Frame(settings_frame2)
        yscale_frame.grid(row=0, column=1, padx=20)
        
        tk.Label(yscale_frame, text="Y Scale:").pack(side=tk.LEFT)
        self.yscale_scale = Scale(yscale_frame, from_=0.1, to=10.0, resolution=0.1, 
                                 orient=HORIZONTAL, length=150, command=self.update_y_scale)
        self.yscale_scale.set(1.0)
        self.yscale_scale.pack(side=tk.LEFT)
        
        # Frame for displaying plots
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create figure
        self.fig = Figure(figsize=(8, 8), dpi=100)
        
        # Add subplots
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        
        # Set subplot titles
        self.ax1.set_title("Original ECG Signal")
        self.ax2.set_title("ECG Signal After Noise Removal")
        
        # Initialize plot lines
        self.line_raw, = self.ax1.plot([], [], 'b-')
        self.line_filtered, = self.ax2.plot([], [], 'r-')
        
        # Add figure to user interface
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add status information
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to load ECG file")
        status_label = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Progress indicator
        progress_frame = tk.Frame(self.root)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(progress_frame, text="Progress:").pack(side=tk.LEFT)
        self.progress_var = tk.DoubleVar()
        self.progress_scale = Scale(progress_frame, variable=self.progress_var, from_=0, to=100, 
                                  orient=HORIZONTAL, length=850, command=self.seek_position)
        self.progress_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def update_speed(self, val):
        self.animation_speed = float(val)
        
    def update_display_width(self, val):
        self.display_width = int(val)
        # Reset horizontal axis limits
        if self.signal_raw is not None:
            self.update_axes_limits()

    def update_y_scale(self, val):
        self.y_scale = float(val)
        
        
        # Update plot
        self.update_axes_limits()
        self.update_display()
    
    def update_axes_limits(self):
        # Update axis limits based on current position
        if self.time is not None and len(self.time) > 0:
            start_idx = max(0, self.current_index - int(self.display_width * self.sampling_rate))
            end_idx = min(len(self.time), self.current_index + int(0.1 * self.sampling_rate))
            
            # Ensure there is enough data
            if end_idx > start_idx:
                start_time = self.time[start_idx]
                end_time = start_time + self.display_width
                
                self.ax1.set_xlim(start_time, end_time)
                self.ax2.set_xlim(start_time, end_time)
                
                # Update vertical axis limits based on Y scale
                if self.signal_raw is not None and len(self.signal_raw) > 0:
                    data_slice = self.signal_raw[max(0, start_idx):min(len(self.signal_raw), end_idx+1)]
                    if len(data_slice) > 0:
                        center = (np.max(data_slice) + np.min(data_slice)) / 2
                        range_val = (np.max(data_slice) - np.min(data_slice)) / 2
                        min_val = center - (range_val / self.y_scale)
                        max_val = center + (range_val / self.y_scale)
                        self.ax1.set_ylim(min_val, max_val)
                
                if self.signal_filtered is not None and len(self.signal_filtered) > 0:
                    data_slice = self.signal_filtered[max(0, start_idx):min(len(self.signal_filtered), end_idx+1)]
                    if len(data_slice) > 0:
                        center = (np.max(data_slice) + np.min(data_slice)) / 2
                        range_val = (np.max(data_slice) - np.min(data_slice)) / 2
                        min_val = center - (range_val / self.y_scale)
                        max_val = center + (range_val / self.y_scale)
                        self.ax2.set_ylim(min_val, max_val)
    
    def seek_position(self, val):
        if not self.is_playing and self.signal_raw is not None:
            # Convert percentage to index
            pos = int(float(val) / 100 * len(self.signal_raw))
            self.current_index = pos
            self.update_display()
    
    def toggle_live_display(self):
        if self.signal_raw is None:
            messagebox.showwarning("Warning", "Please load an ECG file first")
            return
            
        if self.is_playing:
            # Stop display
            self.is_playing = False
            if self.animation:
                self.animation.event_source.stop()
            self.play_btn.config(text="Start Live Display")
            self.status_var.set("Live display stopped")
        else:
            # Start display
            self.is_playing = True
            self.play_btn.config(text="Stop Live Display")
            self.status_var.set("Live display in progress...")
            
            # Reset index if reached the end
            if self.current_index >= len(self.signal_raw) - 1:
                self.current_index = 0
                
            # Start animation
            self.start_animation()
            
    def start_animation(self):
        # Stop any previous animation
        if self.animation:
            self.animation.event_source.stop()
            
        # Initialize plot axes
        self.update_axes_limits()
        
        # Start new animation
        self.animation = FuncAnimation(
            self.fig, self.update_animation, interval=50, 
            blit=False, cache_frame_data=False
        )
        self.canvas.draw()
    
    def update_animation(self, frame):
        if not self.is_playing:
            return
            
        # Calculate number of points to advance based on speed
        step = int(self.sampling_rate * 0.05 * self.animation_speed)  # 0.05 seconds * speed factor
        
        # Update index
        self.current_index += step
        
        # Check if index has reached the end of data
        if self.current_index >= len(self.signal_raw):
            self.current_index = 0
            
        # Update progress bar
        progress_percent = (self.current_index / len(self.signal_raw)) * 100
        self.progress_var.set(progress_percent)
            
        self.update_display()
        
    def update_display(self):
        # Update axis limits
        self.update_axes_limits()
        
        # Update line data
        visible_start = max(0, self.current_index - int(self.display_width * self.sampling_rate))
        visible_end = self.current_index
        
        if visible_start < visible_end:
            self.line_raw.set_data(self.time[visible_start:visible_end], self.signal_raw[visible_start:visible_end])
            
            if self.signal_filtered is not None:
                self.line_filtered.set_data(self.time[visible_start:visible_end], self.signal_filtered[visible_start:visible_end])
        
        # Update plot
        self.canvas.draw_idle()
    
    def load_file(self):
        # Stop live display
        if self.is_playing:
            self.toggle_live_display()
            
        # Open file selection dialog
        file_path = filedialog.askopenfilename(
            title="Select ECG File",
            filetypes=[
                ("All Supported Files", "*.txt *.csv *.xlsx *.xls *.dat"),
                ("Text Files", "*.txt"),
                ("CSV Files", "*.csv"),
                ("Excel Files", "*.xlsx *.xls"),
                ("DAT Files", "*.dat")
            ]
        )
        
        if not file_path:
            return
            
        try:
            # Determine file type and process it
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.txt' or file_ext == '.dat':
                # Read text or DAT file
                self.signal_raw = np.loadtxt(file_path)
            elif file_ext == '.csv':
                # Read CSV file
                df = pd.read_csv(file_path)
                self.signal_raw = df.iloc[:, 0].values  # Assume signal is in the first column
            elif file_ext == '.xlsx' or file_ext == '.xls':
                # Read Excel file
                df = pd.read_excel(file_path)
                self.signal_raw = df.iloc[:, 0].values  # Assume signal is in the first column
            
            # Save a copy of the original signal
            self.signal_original = self.signal_raw.copy()
            
            # Create time axis
            self.time = np.arange(0, len(self.signal_raw)) / self.sampling_rate
            
            # Initialize plot
            self.current_index = 0
            self.signal_filtered = None
            self.filter_count = 0
            self.filter_count_var.set("Filter Applications: 0")
            
            # Prepare plots
            self.ax1.clear()
            self.ax2.clear()
            
            
            self.ax1.set_title("Original ECG Signal")
            self.ax1.set_xlabel("Time (Seconds)")
            self.ax1.set_ylabel("Voltage")
            self.ax1.grid(True)
            
            self.ax2.set_title("ECG Signal After Noise Removal")
            self.ax2.set_xlabel("Time (Seconds)")
            self.ax2.set_ylabel("Voltage")
            self.ax2.grid(True)
            
            # Reinitialize plot lines
            self.line_raw, = self.ax1.plot([], [], 'b-')
            self.line_filtered, = self.ax2.plot([], [], 'r-')
            
            # Update only the top plot (before applying filter)
            visible_start = 0
            visible_end = min(int(self.display_width * self.sampling_rate), len(self.signal_raw))
            self.line_raw.set_data(self.time[visible_start:visible_end], self.signal_raw[visible_start:visible_end])
            
            # Update axis limits
            self.update_axes_limits()
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Update progress bar
            self.progress_var.set(0)
            
            self.status_var.set(f"File loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error Reading File", str(e))
    
    def process_signal(self):
        if self.signal_raw is None:
            messagebox.showwarning("Warning", "Please load an ECG file first")
            return
        
        try:
            # Ensure the appropriate signal is used for processing
            if self.signal_filtered is None:
                # First processing
                signal_to_process = self.signal_raw
            else:
                # Subsequent processing
                signal_to_process = self.signal_filtered
            
            # Remove baseline wander using a high-pass filter
            b, a = signal.butter(4, 0.5/(self.sampling_rate/2), 'high')
            filtered_signal = signal.filtfilt(b, a, signal_to_process)
            
            # Remove 50 Hz AC noise using a notch filter
            notch_freq = 50  # Can be adjusted to 60 Hz for regions with 60 Hz powerline frequency
            b, a = signal.iirnotch(notch_freq, 30, self.sampling_rate)
            filtered_signal = signal.filtfilt(b, a, filtered_signal)
            
            # Add a low-pass filter to remove high-frequency noise
            b, a = signal.butter(4, 40/(self.sampling_rate/2), 'low')
            filtered_signal = signal.filtfilt(b, a, filtered_signal)
            
            # Update the processed signal
            self.signal_filtered = filtered_signal
            
            # Increment filter application count
            self.filter_count += 1
            self.filter_count_var.set(f"Filter Applications: {self.filter_count}")
            
            self.status_var.set(f"Signal processed successfully - Noise removed {self.filter_count} times")
            
            # Update plot if in stopped mode
            if not self.is_playing:
                self.update_display()
            
        except Exception as e:
            messagebox.showerror("Error in Processing", str(e))
    
    def reset_filter(self):
        """Reset the filter and use the original signal"""
        if self.signal_original is None:
            return
            
        self.signal_raw = self.signal_original.copy()
        self.signal_filtered = None
        self.filter_count = 0
        self.filter_count_var.set("Filter Applications: 0")
        
        # Update plot
        self.update_display()
        self.status_var.set("Filter reset")
    
    def save_filtered_signal(self):
        if self.signal_filtered is None:
            messagebox.showwarning("Warning", "Please process the signal first")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Filtered Signal",
            defaultextension=".csv",
            filetypes=[
                ("CSV File", "*.csv"),
                ("Text File", "*.txt"),
                ("Excel File", "*.xlsx")
            ]
        )
        
        if not file_path:
            return
            
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Create a DataFrame containing time, original signal, and filtered signal
            df = pd.DataFrame({
                'time': self.time,
                'original_signal': self.signal_original,
                'filtered_signal': self.signal_filtered
            })
            
            if file_ext == '.csv':
                df.to_csv(file_path, index=False)
            elif file_ext == '.txt':
                df.to_csv(file_path, sep='\t', index=False)
            elif file_ext == '.xlsx':
                df.to_excel(file_path, index=False)
            
            self.status_var.set(f"Filtered signal saved to: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error Saving File", str(e))

def main():
    root = tk.Tk()
    app = ECGLiveFilterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
