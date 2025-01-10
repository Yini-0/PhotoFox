// MARK: - PhotoSearchApp.swift
import SwiftUI
import SwiftData
import UIKit

class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?
    
    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions:
    [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        ValueTransformer.setValueTransformer(
            MLMultiArrayTransformer(),
            forName: NSValueTransformerName("MLMultiArrayTransformer"))
        return true
    }
}

@main
struct PhotoFoxApp: App {
//    var modelContainer: ModelContainer = {
//        let schema = Schema([PhotoRecord.self])
//        
//        let modelConfiguration = ModelConfiguration(
//            schema: schema,
//            isStoredInMemoryOnly: false
//        )
//        
//        do {
//            return try ModelContainer(
//                for: schema,
//                configurations: [modelConfiguration]
//            )
//        } catch {
//            fatalError("Could not initialize ModelContainer: \(error)")
//        }
//    }()
    
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        //.modelContainer(modelContainer)
    }
}
