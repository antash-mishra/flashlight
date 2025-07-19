import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib, Pango
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
import requests
import webbrowser
from datetime import datetime
import db
import router
import json
import subprocess
import sys
import os
from native_messaging import NativeMessaging

class TabSearchApp:
    def __init__(self):
        
        # Create GTK window
        self.setup_ui()
        
        # Start Flask server in background thread with FAISS storage
        self.router = router.Router()

        self.native_messaging = NativeMessaging()
        

    def setup_ui(self):
        """Create Spotlight-like interface"""
        # Main window - centered, no decorations
        self.window = Gtk.Window()
        self.window.set_decorated(False)  # No title bar
        self.window.set_resizable(False)
        self.window.set_modal(True)
        self.window.set_type_hint(Gdk.WindowTypeHint.DIALOG)
        self.window.set_skip_taskbar_hint(True)
        self.window.set_keep_above(True)
        
        # Center on screen
        self.window.set_position(Gtk.WindowPosition.CENTER)
        self.window.set_default_size(600, 400)
        
        # Connect events
        self.window.connect("destroy", Gtk.main_quit)
        self.window.connect("key-press-event", self.on_key_press)
        self.window.connect("focus-out-event", self.on_focus_out)
        
        # Apply CSS for Spotlight-like styling
        self.apply_spotlight_style()
        
        # Main container with rounded corners effect
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        main_box.get_style_context().add_class("spotlight-container")
        
        # Search section
        search_container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        search_container.get_style_context().add_class("search-container")
        search_container.set_margin_left(20)
        search_container.set_margin_right(20)
        search_container.set_margin_top(20)
        search_container.set_margin_bottom(10)
        
        # Search icon
        search_icon = Gtk.Image.new_from_icon_name("edit-find-symbolic", Gtk.IconSize.LARGE_TOOLBAR)
        search_icon.get_style_context().add_class("search-icon")
        search_container.pack_start(search_icon, False, False, 10)
        
        # Search entry
        self.search_entry = Gtk.Entry()
        self.search_entry.set_placeholder_text("Search tabs...")
        self.search_entry.get_style_context().add_class("spotlight-search")
        self.search_entry.connect("changed", self.on_search_changed)
        self.search_entry.connect("activate", self.on_search_activate)
        self.search_entry.set_has_frame(False)
        search_container.pack_start(self.search_entry, True, True, 0)
        
        main_box.pack_start(search_container, False, False, 0)
        
        # Separator
        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator.get_style_context().add_class("spotlight-separator")
        main_box.pack_start(separator, False, False, 0)
        
        # Results area
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.get_style_context().add_class("results-scroll")
        scrolled.set_margin_left(0)
        scrolled.set_margin_right(0)
        scrolled.set_margin_bottom(0)
        
        self.results_listbox = Gtk.ListBox()
        self.results_listbox.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.results_listbox.connect("row-activated", self.on_tab_selected)
        self.results_listbox.get_style_context().add_class("spotlight-results")
        
        scrolled.add(self.results_listbox)
        main_box.pack_start(scrolled, True, True, 0)
        
        # Status/hint label at bottom
        self.status_label = Gtk.Label()
        self.status_label.set_text("Type to search your browser tabs...")
        self.status_label.get_style_context().add_class("spotlight-hint")
        self.status_label.set_margin_left(20)
        self.status_label.set_margin_right(20)
        self.status_label.set_margin_bottom(15)
        self.status_label.set_xalign(0.5)  # Center horizontally
        main_box.pack_start(self.status_label, False, False, 0)
        
        self.window.add(main_box)
        
        # Initially hide the window
        self.window.hide()
        
        # Set up global hotkey (Ctrl+Space)
        self.setup_hotkey()
    
    def apply_spotlight_style(self):
        """Apply Spotlight-like CSS styling"""
        css_provider = Gtk.CssProvider()
        css_data = """
        .spotlight-container {
            background: rgba(40, 40, 40, 0.95);
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }
        
        .search-container {
            background: transparent;
        }
        
        .spotlight-search {
            background: transparent;
            border: none;
            color: #ffffff;
            font-size: 24px;
            font-weight: 300;
            padding: 8px 0px;
        }
        
        .spotlight-search:focus {
            outline: none;
            box-shadow: none;
        }
        
        .search-icon {
            color: #999999;
            margin-right: 8px;
        }
        
        .spotlight-separator {
            background: rgba(255, 255, 255, 0.1);
            margin: 0px 20px;
        }
        
        .spotlight-results {
            background: transparent;
            border: none;
        }
        
        .spotlight-results row {
            background: transparent;
            border: none;
            padding: 0px;
            margin: 0px 20px;
        }
        
        .spotlight-results row:hover {
            background: rgba(100, 150, 255, 0.2);
            border-radius: 6px;
        }
        
        .spotlight-results row:selected {
            background: rgba(100, 150, 255, 0.3);
            border-radius: 6px;
        }
        
        .result-title {
            color: #ffffff;
            font-size: 16px;
            font-weight: 500;
        }
        
        .result-url {
            color: #999999;
            font-size: 13px;
        }
        
        .result-score {
            color: #666666;
            font-size: 11px;
        }
        
        .spotlight-hint {
            color: #666666;
            font-size: 12px;
        }
        
        .results-scroll {
            background: transparent;
        }
        """
        
        css_provider.load_from_data(css_data.encode())
        screen = Gdk.Screen.get_default()
        style_context = Gtk.StyleContext()
        style_context.add_provider_for_screen(screen, css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)


    def setup_hotkey(self):
        """Set up global hotkey to show/hide the window"""
        # This is a simplified version - for true global hotkeys, you'd need additional libraries
        # For now, we'll show the window on startup
        print("Press Ctrl+Space to show/hide the search window")
        print("Note: Global hotkeys require additional setup - window will show automatically")
        
        # Auto-show for testing
        GLib.timeout_add(1000, self.show_spotlight)
    
    def show_spotlight(self):
        """Show the Spotlight-like search window"""
        self.window.show_all()
        self.window.present()
        self.search_entry.grab_focus()
        return False  # Don't repeat the timeout
    
    def hide_spotlight(self):
        """Hide the search window"""
        self.window.hide()
        self.search_entry.set_text("")  # Clear search
        self.clear_results()
    
    def toggle_spotlight(self):
        """Toggle visibility of the search window"""
        if self.window.get_visible():
            self.hide_spotlight()
        else:
            self.show_spotlight()
    
    def on_key_press(self, widget, event):
        """Handle key press events"""
        keyname = Gdk.keyval_name(event.keyval)
        
        if keyname == "Escape":
            self.hide_spotlight()
            return True
        
        # Arrow key navigation
        if keyname == "Down":
            self.navigate_results(1)
            return True
        elif keyname == "Up":
            self.navigate_results(-1)
            return True
        elif keyname == "Return":
            self.activate_selected_result()
            return True
            
        return False
    
    def on_focus_out(self, widget, event):
        """Hide when focus is lost"""
        # Uncomment to hide on focus loss (like real Spotlight)
        # self.hide_spotlight()
        return False
    
    def navigate_results(self, direction):
        """Navigate through search results with arrow keys"""
        selected = self.results_listbox.get_selected_row()
        if not selected:
            # Select first item
            first_row = self.results_listbox.get_row_at_index(0)
            if first_row:
                self.results_listbox.select_row(first_row)
            return
        
        current_index = selected.get_index()
        new_index = current_index + direction
        
        # Wrap around
        row_count = len(self.results_listbox.get_children())
        if new_index < 0:
            new_index = row_count - 1
        elif new_index >= row_count:
            new_index = 0
        
        new_row = self.results_listbox.get_row_at_index(new_index)
        if new_row:
            self.results_listbox.select_row(new_row)
    
    def activate_selected_result(self):
        """Activate the selected result"""
        selected = self.results_listbox.get_selected_row()
        if selected:
            self.on_tab_selected(self.results_listbox, selected)
    
    def clear_results(self):
        """Clear search results"""
        for child in self.results_listbox.get_children():
            self.results_listbox.remove(child)



    def on_search(self, widget):
        """Handle search button click"""
        query = self.search_entry.get_text()
        if not query:
            return
        
        # Clear previous results
        for child in self.results_listbox.get_children():
            self.results_listbox.remove(child)
        
        # Trigger embedding update if needed
        if not self.router.tab_embeddings:
            self.router.trigger_embedding_update()
            self.update_status("Processing embeddings...")
            return
        
        # Perform search
        results = self.router.search_tabs(query)
        
        if not results:
            self.update_status("No results found")
            return
        
        # Display results
        for result in results:
            self.add_result_row(result)
        
        self.update_status(f"Found {len(results)} results")

    def on_search_changed(self, widget):
        """Handle search text change - real-time search"""
        query = self.search_entry.get_text()
        
        # Clear previous results
        self.clear_results()
        
        if not query:
            self.update_status("Type to search your browser tabs...")
            return
        
        # Trigger embedding update if needed
        if not self.router.tab_embeddings:
            self.router.trigger_embedding_update()
            self.update_status("Processing embeddings...")
            return
        
        # Perform search
        results = self.router.search_tabs(query)
        
        if not results:
            self.update_status("No results found")
            return
        
        # Display results
        for result in results:
            self.add_result_row(result)
        
        # Auto-select first result
        first_row = self.results_listbox.get_row_at_index(0)
        if first_row:
            self.results_listbox.select_row(first_row)
        
        self.update_status(f"{len(results)} results")

    def on_search_activate(self, widget):
        """Handle Enter key in search - activate selected result"""
        self.activate_selected_result()


    def on_refresh(self, widget):
        """Handle refresh button click"""
        self.update_status("Requesting tab sync from browser...")
        # The browser extension should call /update-tabs when it receives this
        # For now, just update the status

    def add_result_row(self, result):
        """Add a result row to the listbox - Spotlight style"""
        row = Gtk.ListBoxRow()
        row.get_style_context().add_class("spotlight-result-row")
        
        # Create result display
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        vbox.set_margin_left(15)
        vbox.set_margin_right(15)
        vbox.set_margin_top(8)
        vbox.set_margin_bottom(8)
        
        # Title - use proper GTK markup without CSS classes
        title_label = Gtk.Label()
        title_label.set_markup(f"<span weight='bold' foreground='white'>{result['title']}</span>")
        title_label.get_style_context().add_class("result-title")
        title_label.set_line_wrap(True)
        title_label.set_max_width_chars(60)
        title_label.set_xalign(0)
        title_label.set_ellipsize(Pango.EllipsizeMode.END)
        vbox.pack_start(title_label, False, False, 0)
        
        # URL
        url_label = Gtk.Label()
        url_label.set_text(result['url'])
        url_label.get_style_context().add_class("result-url")
        url_label.set_line_wrap(True)
        url_label.set_max_width_chars(70)
        url_label.set_xalign(0)
        url_label.set_ellipsize(Pango.EllipsizeMode.END)
        vbox.pack_start(url_label, False, False, 0)
        
        # Similarity score (subtle)
        score_label = Gtk.Label()
        score_label.set_text(f"Relevance: {result['hybrid_score']:.1%}")
        score_label.get_style_context().add_class("result-score")
        score_label.set_xalign(0)
        vbox.pack_start(score_label, False, False, 0)
        
        row.add(vbox)
        
        # Store tab data in row
        row.tab_data = result
        
        self.results_listbox.add(row)
        row.show_all()

    def on_tab_selected(self, listbox, row):
        """Handle tab selection"""
        if row and hasattr(row, 'tab_data'):
            tab_data = row.tab_data
            
            # Send message to browser extension to switch to tab
            try:
                success = self.native_messaging.send_message('openTab', {
                    'tab_id': tab_data['tab_id'],
                    'window_id': tab_data['window_id'] 
                })
                
                if success:
                    self.update_status(f"✅ Tab switching request sent: {tab_data['title']}")
                else:
                    self.update_status(f"❌ Failed to send tab switching request: {tab_data['title']}")
  
            except Exception as e:
                print(f"Error switching tab: {e}")
                self.update_status("Error switching tab")


    def update_status(self, message):
        """Update status label"""
        self.status_label.set_text(f"{datetime.now().strftime('%H:%M:%S')} - {message}")
        
    def run(self):
        """Start the GTK main loop"""
        Gtk.main()

if __name__ == "__main__":
    app = TabSearchApp()
    app.run()