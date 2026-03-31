require_relative "boot"

require "rails"
require "active_model/railtie"
require "active_job/railtie"
require "active_record/railtie"
require "active_storage/engine"
require "action_controller/railtie"
require "action_mailer/railtie"
require "action_mailbox/engine"
require "action_text/engine"
require "action_view/railtie"
require "action_cable/engine"
require "rails/test_unit/railtie"

# Require the gems listed in Gemfile, including any gems
# you've limited to :test, :development, or :production.
Bundler.require(*Rails.groups)

module EmailClassifier
  class Application < Rails::Application
    # Initialize configuration defaults for originally generated Rails version.
    config.load_defaults 7.2

    # -- Time zone ------------------------------------------------------------
    config.time_zone = "UTC"
    config.active_record.default_timezone = :utc

    # -- Encoding -------------------------------------------------------------
    config.encoding = "utf-8"

    # -- Generators -----------------------------------------------------------
    config.generators do |g|
      g.test_framework :test_unit, fixture: false
      g.stylesheets false
      g.javascript false
      g.helper false
    end

    # -- Asset pipeline --------------------------------------------------------
    # config.assets.enabled = true

    # -- Autoloading -----------------------------------------------------------
    # Rails 7.2 uses Zeitwerk by default; services dir is autoloaded.
    config.autoload_lib(ignore: %w[assets tasks])

    # -- Middleware ------------------------------------------------------------
    config.middleware.use ActionDispatch::Cookies
    config.middleware.use ActionDispatch::Session::CookieStore

    # -- Logger ----------------------------------------------------------------
    config.log_level = :info
  end
end
