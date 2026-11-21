plugins {
    kotlin("jvm") version "2.4.10"
}

group = "dev.gogamen"
version = "0.1.0"

repositories {
    mavenCentral()
}

// Optional scripting source set (GOG-83 spike). Not part of the main artifact: the main jar
// never depends on kotlin-scripting. Consumers opt in via `scriptingJar` / `scriptingRuntimeClasspath`.
val scripting: SourceSet by sourceSets.creating {
    compileClasspath += sourceSets.main.get().output
    runtimeClasspath += sourceSets.main.get().output
}
val scriptingTest: SourceSet by sourceSets.creating {
    compileClasspath += scripting.output + sourceSets.main.get().output
    runtimeClasspath += scripting.output + sourceSets.main.get().output
}
val scriptingVersion = "2.4.10" // keep aligned with the kotlin("jvm") plugin above

configurations {
    named("scriptingImplementation") { extendsFrom(configurations.implementation.get()) }
    named("scriptingTestImplementation") { extendsFrom(configurations.getByName("scriptingImplementation")) }
    named("scriptingTestRuntimeOnly") { extendsFrom(configurations.getByName("scriptingRuntimeOnly")) }
}

dependencies {
    "scriptingImplementation"("org.jetbrains.kotlin:kotlin-scripting-common:$scriptingVersion")
    "scriptingImplementation"("org.jetbrains.kotlin:kotlin-scripting-jvm:$scriptingVersion")
    "scriptingImplementation"("org.jetbrains.kotlin:kotlin-scripting-jvm-host:$scriptingVersion")
    "scriptingTestImplementation"("org.jetbrains.kotlin:kotlin-test-junit5:$scriptingVersion")
    "scriptingTestImplementation"("org.junit.jupiter:junit-jupiter:5.11.4")
    "scriptingTestRuntimeOnly"("org.junit.platform:junit-platform-launcher:1.11.4")

    testImplementation(kotlin("test"))
    testImplementation(kotlin("compiler-embeddable"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.11.4")
}

tasks.test {
    useJUnitPlatform()
    testLogging {
        events("passed", "skipped", "failed")
    }
}

val scriptingTestTask = tasks.register<Test>("scriptingTest") {
    description = "Runs the optional Kotlin scripting spike tests."
    group = "verification"
    testClassesDirs = scriptingTest.output.classesDirs
    classpath = scriptingTest.runtimeClasspath
    useJUnitPlatform()
    testLogging { events("passed", "skipped", "failed") }
}
tasks.check { dependsOn(scriptingTestTask) }
tasks.test { finalizedBy(scriptingTestTask) }

// Separate jar so the scripting layer can be sized/shipped independently of the main artifact.
tasks.register<Jar>("scriptingJar") {
    archiveClassifier.set("scripting")
    from(scripting.output)
}

tasks.register("printScriptingClasspathSizes") {
    notCompatibleWithConfigurationCache("measurement helper")
    doLast {
        val base = configurations.runtimeClasspath.get().files.map { it.name }.toSet()
        val extra = configurations.getByName("scriptingRuntimeClasspath").files.filter { it.name !in base }
        extra.forEach { println("SIZE %10d %s".format(it.length(), it.name)) }
        println("SIZE %10d TOTAL-extra".format(extra.sumOf { it.length() }))
    }
}
