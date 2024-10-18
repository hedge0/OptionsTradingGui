def on_mouse_move(event, plot_manager):
    """
    Handle mouse movement over the plot area.

    Args:
        event: The Matplotlib event triggered by mouse movement.
        plot_manager (PlotManager): The PlotManager instance handling the plot.

    This function updates the coordinates displayed in the UI and, if a mouse button
    is pressed, pans the plot according to the movement.
    """
    if event.inaxes:
        if event.xdata is not None and event.ydata is not None:
            x_coord = f"{event.xdata:.2f}"
            y_coord = f"{event.ydata:.4f}"
            plot_manager.coord_label.config(text=f"X: {x_coord}    Y: {y_coord}")

        if plot_manager.press_event is not None:
            dx = event.xdata - plot_manager.press_event.xdata
            dy = event.ydata - plot_manager.press_event.ydata
            cur_xlim = plot_manager.ax.get_xlim()
            cur_ylim = plot_manager.ax.get_ylim()
            plot_manager.ax.set_xlim(cur_xlim[0] - dx, cur_xlim[1] - dx)
            plot_manager.ax.set_ylim(cur_ylim[0] - dy, cur_ylim[1] - dy)
            plot_manager.canvas.draw()

def on_scroll(event, plot_manager):
    """
    Handle scroll events to zoom in and out of the plot.

    Args:
        event: The Matplotlib event triggered by the scroll wheel.
        plot_manager (PlotManager): The PlotManager instance handling the plot.

    This function adjusts the plot's x and y limits based on the scroll direction,
    effectively zooming in or out centered around the cursor's position.
    """
    if event.xdata is None or event.ydata is None:
        return

    base_scale = 1.2
    cur_xlim = plot_manager.ax.get_xlim()
    cur_ylim = plot_manager.ax.get_ylim()
    if event.button == 'up':
        scale_factor = 1 / base_scale
    elif event.button == 'down':
        scale_factor = base_scale
    else:
        scale_factor = 1
    new_xlim = [event.xdata - (event.xdata - cur_xlim[0]) * scale_factor,
                event.xdata + (cur_xlim[1] - event.xdata) * scale_factor]
    new_ylim = [event.ydata - (event.ydata - cur_ylim[0]) * scale_factor,
                event.ydata + (cur_ylim[1] - event.ydata) * scale_factor]
    plot_manager.ax.set_xlim(new_xlim)
    plot_manager.ax.set_ylim(new_ylim)
    plot_manager.canvas.draw()

def on_press(event, plot_manager):
    """
    Handle mouse press events to start dragging the plot.

    Args:
        event: The Matplotlib event triggered by pressing a mouse button.
        plot_manager (PlotManager): The PlotManager instance handling the plot.

    This function stores the initial position of the mouse press, which is used
    to calculate the displacement during dragging.
    """
    if event.inaxes:
        plot_manager.press_event = event

def on_release(event, plot_manager):
    """
    Handle mouse release events to stop dragging the plot.

    Args:
        event: The Matplotlib event triggered by releasing a mouse button.
        plot_manager (PlotManager): The PlotManager instance handling the plot.

    This function resets the press_event, signaling the end of the dragging operation.
    """
    plot_manager.press_event = None
