import flet as ft
import os
import csv
import sys
import subprocess
import platform
from datetime import datetime
import asyncio
from parse import auto_async, select_folder_and_process, process_folder_async, group_students_by_month, delete_student_from_table, export_month_to_excel, save_preview # NEW FUNCTION NAME


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(HERE, "..")))


def main(page: ft.Page):
    # ── page settings
    page.title = "Document Parser"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor = "#F5F7FA"
    page.window_maximized = True
    page.padding = 30
    page.spacing = 20
    parsed_records = [] 
    selected_folder_info = None   

    # ── overlays & controls
    file_picker = ft.FilePicker()
    file_picker.with_data = True
    page.overlay.append(file_picker)

    # ── Initializations
    status_text = ft.Text("No files selected", size=14, color="#64748B")
    snackbar = ft.SnackBar(content=ft.Text(""))
    page.overlay.append(snackbar)
    table_placeholder = ft.Column([], scroll=ft.ScrollMode.ALWAYS)
    selected_folder_path = None
    current_page = 0

    # ── render table
    def render_table(records: list[dict]):
        parse_btn.disabled = True
        table_placeholder.controls.clear()
        if not records:
            table_placeholder.controls.append(ft.Text("No student data found."))
            page.update()
            return

    # header
        header = ft.Container(
            content=ft.Row([
                ft.Text("Name", expand=2, text_align=ft.TextAlign.CENTER, weight=ft.FontWeight.BOLD, color="#1E293B", size=13),
                ft.Text("DL#", expand=1, text_align=ft.TextAlign.CENTER, weight=ft.FontWeight.BOLD, color="#1E293B", size=13),
                ft.Text("DOB", expand=1, text_align=ft.TextAlign.CENTER, weight=ft.FontWeight.BOLD, color="#1E293B", size=13),
                ft.Text("Skills Test Date", expand=1, text_align=ft.TextAlign.CENTER, weight=ft.FontWeight.BOLD, color="#1E293B", size=13),
                ft.Text("XP", expand=1, text_align=ft.TextAlign.CENTER, weight=ft.FontWeight.BOLD, color="#1E293B", size=13), 
                ft.Text("XF", expand=1, text_align=ft.TextAlign.CENTER, weight=ft.FontWeight.BOLD, color="#1E293B", size=13), 
                ft.Text("ITTD", expand=1, text_align=ft.TextAlign.CENTER, weight=ft.FontWeight.BOLD, color="#1E293B", size=13),
                ft.Text("ITAD", expand=1, text_align=ft.TextAlign.CENTER, weight=ft.FontWeight.BOLD, color="#1E293B", size=13),
                ft.Text("DE-964", expand=1, text_align=ft.TextAlign.CENTER, weight=ft.FontWeight.BOLD, color="#1E293B", size=13),
                ft.Text("ADEE", expand=1, text_align=ft.TextAlign.CENTER, weight=ft.FontWeight.BOLD, color="#1E293B", size=13),
            ]),
            bgcolor="#E2E8F0",
            padding=15,
            border_radius=8,
        )
        table_placeholder.controls.append(header)

        # rows - THIS NEEDS TO BE INDENTED INSIDE THE FUNCTION
        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                continue
        
        row_bg = "#FFFF" if idx % 2 == 0 else "#F8FAFC"
        row = ft.Container(
            content=ft.Row([
                ft.Text(record.get("name", "N/A"), text_align=ft.TextAlign.LEFT, expand=2, color="#334155", size=12),
                ft.Text(record.get("dl_number", "N/A"), text_align=ft.TextAlign.CENTER, expand=1, color="#334155", size=12),
                ft.Text(record.get("dob", "N/A"), text_align=ft.TextAlign.CENTER, expand=1, color="#334155", size=12),
                ft.Text(record.get("skills_test_date", "N/A"), text_align=ft.TextAlign.CENTER, expand=1, color="#334155", size=12),
                ft.Text(record.get("xp", "N/A"),text_align=ft.TextAlign.CENTER, expand=1, color="#334155", size=12), 
                ft.Text(record.get("xf", "N/A"),text_align=ft.TextAlign.CENTER, expand=1, color="#334155", size=12), 
                ft.Text(record.get("ittd_completion_date", "N/A"), text_align=ft.TextAlign.CENTER, expand=1, color="#334155", size=12),
                ft.Text(record.get("itad_completion_date", "N/A"), text_align=ft.TextAlign.CENTER, expand=1, color="#334155", size=12),
                ft.Text(record.get("de_964_number", "N/A"), text_align=ft.TextAlign.CENTER, expand=1, color="#334155", size=12),
                ft.Text(record.get("adee_number", "N/A"), text_align=ft.TextAlign.CENTER, expand=1, color="#334155", size=12),
            ]),
            bgcolor=row_bg,
            padding=12,
            border_radius=6,
        )
        table_placeholder.controls.append(row)
    
    page.update()  # THIS ALSO NEEDS TO BE INSIDE THE FUNCTION

    # ── event handlers
    def on_files_selected(e: ft.FilePickerResultEvent):
        if not e.files:
            status_text.value = "⚠️ No files selected"
            status_text.update()
            return

        # Use dat folder in src directory
        dat_dir = os.path.join(HERE, "dat")  # Changed from "..", "dat" to just "dat"
        os.makedirs(dat_dir, exist_ok=True)

        uploaded = []
        for f in e.files:
            try:
                src = f.path
                dst = os.path.join(dat_dir, f.name)
                with open(src, "rb") as s, open(dst, "wb") as d:
                    d.write(s.read())
                uploaded.append(f.name)
            except Exception as err:
                print(f"❌ Failed to upload {f.name}: {err}")

        if uploaded:
            status_text.value = f"✅ Uploaded: {', '.join(uploaded)}"
        else:
            status_text.value = "⚠️ Upload failed"
        status_text.update()

    def handle_upload(e: ft.ControlEvent):
        file_picker.pick_files(allow_multiple=True)


    def show_snackbar(message: str, color: str = ft.Colors.GREEN_400):
        snackbar.content.value = message
        snackbar.bgcolor = color
        snackbar.open = True
        page.update()

    def show_processing(message: str, show: bool = True):
        if show:
            processing_indicator.content.controls[1].value = message
            processing_indicator.visible = True
        else:
            processing_indicator.visible = False
        page.update()
    
    def handle_start_parse(e: ft.ControlEvent):
        nonlocal selected_folder_info
        
        # Show persistent processing indicator
        parse_btn.disabled = True
        parse_btn.text = "Processing..."
        
        # Check if we have a selected folder or should use dat folder
        if selected_folder_info:
            file_count = len(selected_folder_info["files"])
            folder_name = os.path.basename(selected_folder_info["folder_path"])
            show_processing(f"🔄 Processing {file_count} files from {folder_name}...")
            status_text.value = f"🔄 Processing {file_count} files from folder..."
        else:
            show_processing("🔄 Processing files from dat folder...")
            status_text.value = "🔄 Starting parse..."
        
        status_text.update()
        
        async def runner():
            nonlocal selected_folder_info
            try:
                if selected_folder_info:
                    # Process folder files with timeout
                    show_processing(f"🔄 Extracting data from folder files...")
                    data = await asyncio.wait_for(
                        process_folder_async(selected_folder_info), 
                        timeout=690.0
                    )
                else:
                    # Process dat folder files with timeout
                    show_processing("🔄 Extracting data from documents...")
                    data = await asyncio.wait_for(
                        auto_async(),
                        timeout=900.0
                    )

                # Update processing message
                show_processing("🔄 Processing results...")

                if data is None:
                    data = []
                    status_text.value = "⚠️ No data returned from parser"
                else:
                    status_text.value = f"🔄 Found {len(data)} records"

                status_text.update()

                parsed_records.clear()
                parsed_records.extend(data)

                status_text.value = "✅ Done parsing."
                status_text.update()

                render_table(parsed_records)

                # Show final success snackbar
                snackbar.content.value = f"✅ Successfully parsed {len(data)} records!"
                snackbar.bgcolor = ft.Colors.GREEN_400
                snackbar.open = True
                page.update()

            except asyncio.TimeoutError:
                status_text.value = "⏱️ Parsing timed out after 30 seconds"
                status_text.update()
                snackbar.content.value = "⏱️ Timeout: No files found or parsing took too long"
                snackbar.bgcolor = ft.Colors.ORANGE_400
                snackbar.open = True
                page.update()
            except Exception as ex:
                status_text.value = f"❌ Error: {str(ex)}"
                status_text.update()
                snackbar.content.value = f"❌ Error: {str(ex)}"
                snackbar.bgcolor = ft.Colors.RED_400
                snackbar.open = True
                page.update()
                print(f"Parse error: {ex}")
            finally:
                # Hide processing indicator and re-enable button
                show_processing("", False)
                parse_btn.disabled = False
                parse_btn.text = "Start Parse"

                # Clear selected folder after processing
                selected_folder_info = None
                page.update()
        page.run_task(runner)

    def handle_select_folder(e: ft.ControlEvent):
        async def folder_runner():
            nonlocal selected_folder_info
            try:
                folder_info = select_folder_and_process()
                
                if not folder_info:
                    status_text.value = "⚠️ No valid folder selected"
                    status_text.update()
                    snackbar.content.value = "⚠️ No folder selected"
                    snackbar.bgcolor = ft.Colors.ORANGE_400
                    snackbar.open = True
                    page.update()
                    return
                
                selected_folder_info = folder_info
                
                folder_path = folder_info["folder_path"]
                file_count = len(folder_info["files"])
                
                status_text.value = f"📂 Selected folder: {os.path.basename(folder_path)} ({file_count} files) - Click 'Start Parse' to process"
                status_text.update()
                
                snackbar.content.value = f"✅ Folder selected with {file_count} files!"
                snackbar.bgcolor = ft.Colors.GREEN_400
                snackbar.open = True
                page.update()
                
            except Exception as ex:
                status_text.value = f"❌ Error selecting folder: {str(ex)}"
                status_text.update()
                snackbar.content.value = f"❌ Error: {str(ex)}"
                snackbar.bgcolor = ft.Colors.RED_400
                snackbar.open = True
                page.update()
        page.run_task(folder_runner)
    
    def copyt(e):
        if not table_placeholder.controls:
            show_snackbar("No table data to copy", ft.Colors.RED_400)
            return
    
        # Header
        lines = ["Name\tDL#\tDOB\tSkills Test Date\tExam Result\tITTD\tITAD\tDE-964\tADEE"]
        for record in parsed_records:
            values = [
                record.get("name", "N/A"),
                record.get("dl_number", "N/A"),
                record.get("dob", "N/A"),
                record.get("skills_test_date", "N/A"),
                record.get("xf", "N/A"),
                record.get("xp", "N/A"),
                record.get("ittd_completion_date", "N/A"),
                record.get("itad_completion_date", "N/A"),
                record.get("de_964_number", "N/A"),
                record.get("adee_number", "N/A"),
            ]
            lines.append("\t".join(values))
        table_str = "\n".join(lines)
        page.set_clipboard(table_str)
        show_snackbar("Table copied to clipboard!", ft.Colors.GREEN_400)

       
    def clrt(e):
        """Clear only the preview table"""
        table_placeholder.controls.clear()
        parsed_records.clear()
        show_snackbar("Preview cleared", ft.Colors.ORANGE_400)
        page.update()

    def add_preview_to_table():
        """
        Add preview records to the main monthly table.
        FIXED: Now calls the new 'save_preview' persistence hook.
        """
        if not parsed_records:
            show_snackbar("No preview data to add", ft.Colors.RED_400)
            return
        
        # FIXED: Use the new 'save_preview' function (which calls persist_table)
        added_count = save_preview(parsed_records)
        
        if added_count > 0:
            show_snackbar(f"✅ Added {added_count} records to monthly tables!", ft.Colors.GREEN_400)
            
            # Clear preview after successful addition
            table_placeholder.controls.clear()
            parsed_records.clear()
            page.update()

            tables_layout.controls.clear() 
            tables_layout.controls.extend(build_tables_list().controls) 
        else:
            show_snackbar("⚠️ No valid records to add", ft.Colors.ORANGE_400)

    def on_search_change(e):
        query = e.control.value.strip().lower()
        filtered = [
            r for r in parsed_records
            if any(query in str(val).lower() for val in r.values())
        ]
        render_table(filtered)

    def handle_view_files(e):
        folder = selected_folder_path or "./dat"
        abs_folder = os.path.abspath(folder)
        try:

            system = platform.system()
            if system == "Linux":
                subprocess.Popen(['xdg-open', abs_folder])
            elif system == "Darwin":  # macOS
                subprocess.Popen(['open', abs_folder])
            elif system == "Windows":
                os.startfile(abs_folder)
            else:
                raise Exception(f"Unsupported OS: {system}")
            
            show_snackbar(f"✅ Opened folder: {os.path.basename(abs_folder)}", ft.Colors.GREEN_400)
        
        except Exception as ex:
            show_snackbar(f"❌ Failed to open folder: {ex}", ft.Colors.RED_400)
    
    def export_table(e):
        if not parsed_records:
            show_snackbar("No table data to export", ft.Colors.RED_400)
            return
    
        # Create exports folder if it doesn't exist
        exports_dir = os.path.abspath(os.path.join(HERE, "..", "exports"))
        os.makedirs(exports_dir, exist_ok=True)
    
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"student_table_{timestamp}.csv"
        filepath = os.path.join(exports_dir, filename)
    
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ["name", "dl_number", "dob", "skills_test_date",
                             "ittd_completion_date", "itad_completion_date", "de_964_number", "adee_number"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for record in parsed_records:
                    writer.writerow(record)
            show_snackbar(f"✅ Table exported to {filename}", ft.Colors.GREEN_400)
            # Open the exports folder
            try:
                os.startfile(exports_dir)  # Windows
            except:
                pass
        except Exception as ex:
            show_snackbar(f"❌ Export failed: {str(ex)}", ft.Colors.RED_400)

    

    # hook picker result
    file_picker.on_result = on_files_selected

    # ── build UI
    def on_navigation_change(e):
        nonlocal current_page
        current_page = e.control.selected_index

        # Clear and rebuild content
        page.controls.clear()

        if current_page == 0:  # Home
            page.add(home_layout)
        elif current_page == 1:  # Tables
            # FIX: Clear and rebuild tables_layout from fresh data every time
            tables_layout.controls.clear()
            tables_layout.controls.extend(build_tables_list().controls)
            page.add(tables_layout)
        elif current_page == 2:  # Settings
            page.add(settings_layout)

        page.add(nav_bar)  # Always show nav bar
        page.update()
    
    
    upload_btn = ft.Container(
        content=ft.Column([
            ft.Icon(ft.Icons.UPLOAD_FILE, size=48, color="#3B82F6"),
            ft.Text("Upload Files", size=16, weight=ft.FontWeight.W_500, color="#1E293B")
        ], alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        border=ft.border.all(2, "#E2E8F0"),
        border_radius=16,
        height=160, 
        width=250,
        bgcolor="#FFFFFF",
        alignment=ft.alignment.center,
        on_click=handle_upload,
        on_hover=lambda e: setattr(e.control, 'bgcolor', '#F8FAFC' if e.data == "true" else '#FFFFFF') or page.update(),
        shadow=ft.BoxShadow(
            spread_radius=0,
            blur_radius=10,
            color=ft.Colors.with_opacity(0.08, "#000000"),
            offset=ft.Offset(0, 2),
        )
    )

    folder_btn = ft.Container(
        content=ft.Column([
            ft.Icon(ft.Icons.FOLDER_OPEN, size=48, color="#8B5CF6"),
            ft.Text("Select Folder", size=16, weight=ft.FontWeight.W_500, color="#1E293B")
        ], alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        border=ft.border.all(2, "#E2E8F0"),
        border_radius=16,
        height=160,
        width=250,
        bgcolor="#FFFFFF",
        alignment=ft.alignment.center,
        on_click=handle_select_folder,
        on_hover=lambda e: setattr(e.control, 'bgcolor', '#F8FAFC' if e.data == "true" else '#FFFFFF') or page.update(),
        shadow=ft.BoxShadow(
            spread_radius=0,
            blur_radius=10,
            color=ft.Colors.with_opacity(0.08, "#000000"),
            offset=ft.Offset(0, 2),
        )
    )

    parse_btn = ft.ElevatedButton(
        text="Start Parse",
        on_click=handle_start_parse,
        expand=True,
        bgcolor="#3B82F6",
        color="#FFFFFF",
        height=50,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=10),
        )
    )
    
    file_btn = ft.ElevatedButton(
        text="View Uploaded Files",
        expand=True,
        on_click=handle_view_files,
        bgcolor="#FFFFFF",
        color="#3B82F6",
        height=50,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=10),
            side=ft.BorderSide(2, "#3B82F6"),
        )
    )

    btn_row = ft.Container(
        content=ft.Row([
            ft.ElevatedButton(
                "Add to Table", 
                expand=True, 
                on_click=lambda e: add_preview_to_table(),
                bgcolor="#10B981",
                color="#FFFFFF",
                height=45,
                style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10))
            ),
            ft.ElevatedButton(
                "Clear Preview", 
                expand=True, 
                on_click=clrt,
                bgcolor="#EF4444",
                color="#FFFFFF",
                height=45,
                style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10))
            ),
        ], alignment=ft.MainAxisAlignment.SPACE_EVENLY, spacing=10),
    )

    search_field = ft.TextField(
        hint_text="Search (Name, DL Number, Etc)",
        on_change=on_search_change,
        border_radius=10,
        prefix_icon=ft.Icons.SEARCH,
        expand=True,
    )

    processing_indicator = ft.Container(
        content=ft.Row([
            ft.ProgressRing(width=20, height=20, stroke_width=2),
            ft.Text("", size=14, color=ft.Colors.BLUE_400)
        ], alignment=ft.MainAxisAlignment.CENTER),
        visible=False,
        padding=10,
        bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.BLUE_400),
        border_radius=10
    )
    # Placeholder layouts for other pages
    # Build tables page with mock data
    # Tables page state
    
    # Tables page state
    current_table_view = {"mode": "list", "selected_month": None}
    
    def build_tables_list():
        """Build the list of months from real data
        
        ENHANCED: Now prominently displays 'Unknown' table for documents without skills_test_date
        """
        monthly_data = group_students_by_month()
        
        if not monthly_data:
            return ft.Column([
                ft.Text("No monthly data available", size=18, color="#64748B"),
                ft.Text("Process some documents first to see monthly tables", size=14, color="#94A3B8")
            ], expand=True)
        
        month_cards = ft.Column([], spacing=15)
        
        # ENHANCED: Separate unknown/unk table from regular monthly tables
        unknown_table = None
        regular_tables = []
        
        for month_data in monthly_data:
            if month_data["month"] == "Unknown":
                unknown_table = month_data
            else:
                regular_tables.append(month_data)
        
        # ENHANCED: Display Unknown table first with special styling if it exists
        if unknown_table:
            students = unknown_table["students"]
            
            def make_unknown_handler(data):
                def handler(e):
                    current_table_view["mode"] = "detail"
                    current_table_view["selected_month"] = data
                    show_table_detail()
                return handler
            
            # Special card for Unknown table with warning color
            unknown_card = ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.HELP_OUTLINE, size=28, color="#FFFFFF"),
                    ft.Column([
                        ft.Text("⚠️ Unknown / No Skills Test Date", size=18, weight=ft.FontWeight.BOLD, color="#FFFFFF"),
                        ft.Text(f"{len(students)} documents without skills test date", size=14, color="#FFFFFF")
                    ], spacing=2)
                ]),
                bgcolor="#F59E0B",  # Orange color for warning
                padding=15,
                border_radius=10,
                on_click=make_unknown_handler(unknown_table),
                ink=True,
                border=ft.border.all(3, "#D97706")  # Darker orange border
            )
            month_cards.controls.append(unknown_card)
            
            # Add separator
            month_cards.controls.append(ft.Divider(height=20, color="#E2E8F0"))
        
        # Display regular monthly tables
        for month_data in regular_tables:
            month = month_data["month"]
            year = month_data["year"]
            students = month_data["students"]
            
            def make_click_handler(data):
                def handler(e):
                    current_table_view["mode"] = "detail"
                    current_table_view["selected_month"] = data
                    show_table_detail()
                return handler
            
            card = ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.CALENDAR_MONTH, size=24, color="#3B82F6"),
                    ft.Text(f"{month} {year} ({len(students)} students)", size=18, weight=ft.FontWeight.BOLD)
                ]),
                bgcolor="#E0E7FF",  # Light blue for regular tables
                padding=15,
                border_radius=10,
                on_click=make_click_handler(month_data),
                ink=True,
                border=ft.border.all(1, "#C7D2FE")
            )
            month_cards.controls.append(card)
        
        return ft.Column([
            ft.Text("Monthly Tables", size=24, weight=ft.FontWeight.BOLD),
            ft.TextField(hint_text="Search...", prefix_icon=ft.Icons.SEARCH, border_radius=10),
            month_cards
        ], expand=True, scroll=ft.ScrollMode.ALWAYS)
    
    def build_table_detail(month_data):
        """Build the full table view for a specific month"""
        month = month_data["month"]
        year = month_data["year"]
        students = month_data["students"]
        
        # Table header
        table_header = ft.Container(
            content=ft.Row([
                ft.Text("Name", expand=2, weight=ft.FontWeight.BOLD, size=12),
                ft.Text("DL#", expand=1, weight=ft.FontWeight.BOLD, size=12),
                ft.Text("DOB", expand=1, weight=ft.FontWeight.BOLD, size=12),
                ft.Text("Skills Test Date", expand=1, weight=ft.FontWeight.BOLD, size=12),
                ft.Text("XP", expand=1, weight=ft.FontWeight.BOLD, size=12),
                ft.Text("XF", expand=1, weight=ft.FontWeight.BOLD, size=12),
                ft.Text("ITTD", expand=1, weight=ft.FontWeight.BOLD, size=12),
                ft.Text("ITAD", expand=1, weight=ft.FontWeight.BOLD, size=12),
                ft.Text("DE-964", expand=1, weight=ft.FontWeight.BOLD, size=12),
                ft.Text("ADEE", expand=1, weight=ft.FontWeight.BOLD, size=12),
                ft.Text("", width=50),
            ]),
            bgcolor="#E2E8F0",
            padding=10,
            border_radius=8
        )
        
        # Student rows
        student_rows_container = ft.Column([], spacing=5)
        
        for idx, student in enumerate(students):
            row_bg = "#FFFFFF" if idx % 2 == 0 else "#F8FAFC"
            
           
            def make_delete_handler(name_to_delete):
                def handler(e):
                    if delete_student_from_table(name_to_delete): # Use the new variable
                        show_snackbar(f"✅ Deleted {name_to_delete}", ft.Colors.RED_400)
                        
                        # Update the current view list using name_to_delete
                        current_table_view["selected_month"] = {
                            "month": month,
                            "year": year,
                            "students": [s for s in students if s["name"] != name_to_delete]
                        }
                        show_table_detail()
                    
                    else:
                        show_snackbar(f"⚠️ Failed to delete {name_to_delete} (Not found in table)", ft.Colors.RED_400)
                return handler
            
            ft.IconButton(
                icon=ft.Icons.DELETE_OUTLINE, 
                icon_color="#EF4444", 
                width=50,
                on_click=make_delete_handler(student["name"])
            ),
            
            row = ft.Container(
                content=ft.Row([
                    ft.Text(student.get("name", "N/A"), expand=2, size=11),
                    ft.Text(student.get("dl_number", "N/A"), expand=1, size=11),
                    ft.Text(student.get("dob", "N/A"), expand=1, size=11),
                    ft.Text(student.get("skills_test_date", "N/A"), expand=1, size=11),
                    ft.Text(student.get("xp", "N/A"), expand=1, size=11),
                    ft.Text(student.get("xf", "N/A"), expand=1, size=11),
                    ft.Text(student.get("ittd_completion_date", "N/A"), expand=1, size=11),
                    ft.Text(student.get("itad_completion_date", "N/A"), expand=1, size=11),
                    ft.Text(student.get("de_964_number", "N/A"), expand=1, size=11),
                    ft.Text(student.get("adee_number", "N/A"), expand=1, size=11),
                    ft.IconButton(
                        icon=ft.Icons.DELETE_OUTLINE, 
                        icon_color="#EF4444", 
                        width=50,
                        on_click=make_delete_handler(student["name"])
                    ),
                ]),
                bgcolor=row_bg,
                padding=10,
                border_radius=6
            )
            student_rows_container.controls.append(row)
        
        # Action buttons
        def copy_table_handler(e):
            lines = ["Name\tDL#\tDOB\tSkills Test\tResult\tITTD\tITAD\tDE-964\tADEE"]
            for student in students:
                line = f"{student['name']}\t{student['dl_number']}\t{student['dob']}\t{student['skills_test_date']}\t{student['xp']}\t{student['xf']}\t{student['ittd_completion_date']}\t{student['itad_completion_date']}\t{student['de_964_number']}\t{student['adee_number']}"
                lines.append(line)
            table_str = "\n".join(lines)
            page.set_clipboard(table_str)
            show_snackbar(f"Copied {month} {year} table!", ft.Colors.GREEN_400)
        
        def export_table_handler(e):
            filename = export_month_to_excel(month_data)
            if filename:
                show_snackbar(f"Exported to {filename}", ft.Colors.GREEN_400)
            else:
                show_snackbar("Export failed", ft.Colors.RED_400)
        
        action_btns = ft.Row([
            ft.ElevatedButton(
                "Copy Table", 
                icon=ft.Icons.COPY, 
                bgcolor="#10B981", 
                color="#FFFFFF",
                on_click=copy_table_handler
            ),
            ft.ElevatedButton(
                "Extract Table", 
                icon=ft.Icons.DOWNLOAD, 
                bgcolor="#F59E0B", 
                color="#FFFFFF",
                on_click=export_table_handler
            ),
        ], spacing=10)
        
        def go_back(e):
            current_table_view["mode"] = "list"
            current_table_view["selected_month"] = None
            show_tables_list()
        
        return ft.Column([
            ft.Row([
                ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=go_back, icon_size=24),
                ft.Text(f"{month} {year}", size=24, weight=ft.FontWeight.BOLD)
            ]),
            table_header,
            ft.Container(
                content=student_rows_container,
                expand=True
            ),
            action_btns
        ], expand=True, scroll=ft.ScrollMode.ALWAYS)
    
    def show_tables_list():
        """Switch to list view"""
        page.controls.clear()
        if current_page == 1:
            tables_layout.controls.clear()
            tables_layout.controls.extend(build_tables_list().controls)
            page.add(tables_layout)
            page.add(nav_bar)
        page.update()
    

    def show_table_detail():
        """Switch to detail view and update data"""
        page.controls.clear()

        if current_table_view["selected_month"]:
        
            # FIX: Fetch the latest grouped data from parse.py for accuracy
            all_monthly_data = group_students_by_month()
        
            # Find the latest version of the currently selected month
            current_month_name = current_table_view["selected_month"]["month"]
            current_year = current_table_view["selected_month"]["year"]
        
            latest_month_data = next((
                m for m in all_monthly_data 
                if m["month"] == current_month_name and m["year"] == current_year
            ), current_table_view["selected_month"])
        
            page.add(build_table_detail(latest_month_data))
            page.add(nav_bar)
        page.update()

    
    tables_layout = ft.Column([], expand=True)

    settings_layout = ft.Column([
        ft.Text("Settings Page - Coming Soon", size=24)
    ], expand=True)

    # ── Page Layout
    home_layout = ft.Column([
        ft.Row([
            ft.Container(upload_btn, padding=10, expand=True),
            ft.Container(folder_btn, padding=10, expand=True),
        ], spacing=20),

        ft.Container(
            content=status_text,
            alignment=ft.alignment.center
        ),

        ft.Container(
            content=ft.Row([parse_btn, file_btn]),
        ),

        # Add the persistent processing indicator here
        processing_indicator,

        ft.Container(
            content=btn_row,
        ),

        ft.Container(
            content=search_field,
        ),
        
        ft.Container(
            content=table_placeholder,
            alignment=ft.alignment.top_left,
            expand=True
        ),
    ], expand=True)

    nav_bar = ft.NavigationBar(
        destinations=[
            ft.NavigationBarDestination(icon=ft.Icons.HOME, label="Home"),
            ft.NavigationBarDestination(icon=ft.Icons.TABLE_CHART, label="Tables"),
            ft.NavigationBarDestination(icon=ft.Icons.SETTINGS, label="Settings"),
        ],
    on_change=on_navigation_change,
    )
    
    page.add(home_layout)
    page.add(nav_bar)

ft.app(target=main)
