terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ── Artifact Registry — stores Docker images ──────────────────────────────────

resource "google_artifact_registry_repository" "bot" {
  repository_id = "amazon-device-support-bot"
  format        = "DOCKER"
  location      = var.region
  description   = "Docker images for the Kindle support bot"
}

# ── Secret Manager — production secrets ───────────────────────────────────────

resource "google_secret_manager_secret" "openai_api_key" {
  secret_id = "openai-api-key"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "jwt_secret" {
  secret_id = "jwt-secret"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "langchain_api_key" {
  secret_id = "langchain-api-key"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "postgres_dsn" {
  secret_id = "postgres-dsn"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "mongodb_uri" {
  secret_id = "mongodb-uri"
  replication {
    auto {}
  }
}

# ── Service account for Cloud Run ─────────────────────────────────────────────

resource "google_service_account" "cloud_run_sa" {
  account_id   = "kindle-bot-runner"
  display_name = "Kindle Bot Cloud Run Service Account"
}

# Allow Cloud Run service account to read secrets
resource "google_project_iam_member" "secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# ── Cloud Run service ─────────────────────────────────────────────────────────

locals {
  image = "${var.region}-docker.pkg.dev/${var.project_id}/amazon-device-support-bot/app:${var.image_tag}"
}

resource "google_cloud_run_v2_service" "bot" {
  name     = "amazon-device-support-bot"
  location = var.region

  template {
    service_account = google_service_account.cloud_run_sa.email

    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    containers {
      image = local.image

      resources {
        limits = {
          cpu    = "1"
          memory = "4Gi"
        }
      }

      # Secrets injected as environment variables from Secret Manager
      env {
        name = "OPENAI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.openai_api_key.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "JWT_SECRET"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.jwt_secret.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "LANGCHAIN_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.langchain_api_key.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "POSTGRES_DSN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.postgres_dsn.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "MONGODB_URI"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.mongodb_uri.secret_id
            version = "latest"
          }
        }
      }

      # Non-secret env vars
      env {
        name  = "LANGCHAIN_TRACING_V2"
        value = "true"
      }

      env {
        name  = "GPTCACHE_URL"
        value = ""
      }
    }
  }

  depends_on = [
    google_project_iam_member.secret_accessor,
  ]
}

# Allow unauthenticated requests (public API)
resource "google_cloud_run_v2_service_iam_member" "public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.bot.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
