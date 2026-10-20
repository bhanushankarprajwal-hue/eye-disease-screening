"""Batch screening: many photos sorted by urgency (logic in Code/triage.py, UI in Code/batch_ui.py)."""
import batch_ui
import ui

ui.page_header("Batch screening",
               "Screen many fundus photos at once and sort them by how urgently a doctor should see them.", "layers")
config, model, device = ui.load_or_stop(ui.multiclass_model)
batch_ui.render(config, model, device)
ui.footer()
