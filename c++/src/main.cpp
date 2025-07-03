#include <app.h>

int main(int argc, char **argv) {
    try {
        // Initialize the application
        std::cout << "🚀 Starting the application..." << std::endl;
        Config::LoadConfig("config.json");
        std::cout << "📄 Configuration loaded successfully." << std::endl
                  << "📦 Dependencies initialized." << std::endl;
        App app;
        return app.runApp(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "❌ Exception: " << e.what() << std::endl;
        return 1;
    }
}