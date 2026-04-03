Rails.application.routes.draw do
  root "dashboard#index"

  resources :classifications, only: [:new, :create, :show, :index, :destroy]

  get "dashboard", to: "dashboard#index"
  get "health",    to: "dashboard#health"

  # Health check for load balancers / Docker
  get "up" => "rails/health#show", as: :rails_health_check
end
