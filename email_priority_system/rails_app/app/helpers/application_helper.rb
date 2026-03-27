module ApplicationHelper
  # Returns a Bootstrap badge HTML span for a priority string.
  # e.g. priority_badge("critical") => <span class="badge bg-danger">Red Critical</span>
  def priority_badge(priority, size: nil)
    return content_tag(:span, "Unknown", class: "badge bg-secondary") if priority.blank?

    color   = EmailClassification::PRIORITY_COLORS[priority] || "secondary"
    icon    = EmailClassification::PRIORITY_ICONS[priority]  || ""
    label   = priority.capitalize

    size_class = size == :large ? " fs-6 px-3 py-2" : ""
    content_tag(:span, "#{icon} #{label}".html_safe,
                class: "badge bg-#{color}#{size_class}",
                "aria-label": "Priority: #{label}")
  end

  # Returns a Bootstrap progress bar for a confidence percentage (0..100).
  def confidence_bar(confidence_percent, color: "primary")
    pct = confidence_percent.to_f.clamp(0, 100)
    content_tag(:div, class: "progress", style: "height: 20px;") do
      content_tag(:div, "#{pct.round(1)}%",
                  class: "progress-bar bg-#{color}",
                  role: "progressbar",
                  style: "width: #{pct}%",
                  "aria-valuenow": pct,
                  "aria-valuemin": 0,
                  "aria-valuemax": 100)
    end
  end

  # Returns a human-readable processing time string.
  def format_processing_time(ms)
    return "N/A" unless ms
    ms < 1000 ? "#{ms} ms" : "#{"%.2f" % (ms / 1000.0)} s"
  end

  # Returns the Bootstrap colour name for a given accuracy float (0..1).
  def accuracy_color(accuracy)
    case accuracy.to_f
    when 0.85..  then "success"
    when 0.75..  then "primary"
    when 0.65..  then "warning"
    else               "danger"
    end
  end

  # Render a coloured API status pill.
  def api_status_badge(healthy)
    if healthy
      content_tag(:span, "* Online", class: "badge bg-success")
    else
      content_tag(:span, "* Offline", class: "badge bg-danger")
    end
  end

  # Truncate a string and append ellipsis.
  def short_text(str, length = 80)
    return "" if str.blank?
    str.truncate(length)
  end
end
